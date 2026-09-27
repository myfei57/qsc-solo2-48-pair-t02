"""玻璃液面双路测量与互相校验。

液面只有一路测量时，探头漂移或结料会直接传导成投料门控的误判（跳变一次
就白停一次料）。这里保留原来的接触式探针（probe），再并一路不同原理的
非接触雷达（radar），两路互相校验：

- 两路读数差不超过一致判据 → 采信（取两路均值）；
- 一路坏（变送器自诊断置位、读数越出物理可信范围、读数连续冻结疑似结料）
  → 降级用剩下那路，并明确记录哪一路不可信；
- 两路都"健康"却互相打架 → 不自动投料：挂起持留，等人到现场核实后放行。

放行只针对当前这一轮分歧：双路重新一致后持留自动解除、放行作废，下一次
分歧必须重新到现场核实。出料或清炉后确认凭据与放行一并作废。
"""

from __future__ import annotations

import math
import threading
from dataclasses import dataclass

from ..clock import Clock
from ..config import FurnaceLimits
from ..errors import InterlockError, SequenceError
from ..eventbus import Event, EventBus

#: 液面测量通道：probe 为接触式探针，radar 为不同原理的非接触雷达。
CHANNEL_PRINCIPLE: dict[str, str] = {
    "probe": "接触式探针",
    "radar": "非接触雷达",
}


@dataclass
class _ChannelState:
    """一路液面测量的运行状态。"""

    name: str
    value_mm: float
    updated_at: float
    fault: str | None = None
    offset_mm: float = 0.0
    last_raw: float | None = None
    same_count: int = 0

    def write(self, value_mm: float, at: float) -> None:
        raw = float(value_mm)
        if self.last_raw is not None and raw == self.last_raw:
            self.same_count += 1
        else:
            self.same_count = 1
        self.last_raw = raw
        self.value_mm = raw
        self.updated_at = at


@dataclass(frozen=True)
class Arbitration:
    """一次双路仲裁的结论。"""

    status: str  # agreed / degraded / disagree / no_signal
    value_mm: float  # 采信值；disagree／no_signal 时为保守高值
    trusted: str  # both / probe / radar / none
    untrusted: tuple[str, ...]
    diff_mm: float
    reasons: dict[str, str]


@dataclass(frozen=True)
class LevelCheck:
    """一次液面确认的结果。"""

    level_mm: float
    target_mm: float
    band_mm: float
    ok: bool
    high: bool
    epoch: int
    checked_at: float
    status: str
    channel: str
    untrusted: str | None
    diff_mm: float
    hold: bool
    released: bool
    note: str | None

    def as_dict(self) -> dict:
        return {
            "level_mm": round(self.level_mm, 3),
            "target_mm": self.target_mm,
            "band_mm": self.band_mm,
            "ok": self.ok,
            "high": self.high,
            "epoch": self.epoch,
            "checked_at": round(self.checked_at, 3),
            "status": self.status,
            "channel": self.channel,
            "untrusted": self.untrusted,
            "diff_mm": round(self.diff_mm, 3),
            "hold": self.hold,
            "released": self.released,
            "note": self.note,
        }


class GlassLevel:
    """窑内液面双路测量、仲裁与确认缓存。"""

    def __init__(self, clock: Clock, bus: EventBus, limits: FurnaceLimits) -> None:
        self._clock = clock
        self._bus = bus
        self._limits = limits
        now = self._clock.now()
        self._process_mm = limits.level_target_mm
        self._channels = {
            name: _ChannelState(name=name, value_mm=limits.level_target_mm, updated_at=now)
            for name in CHANNEL_PRINCIPLE
        }
        self._rejected: dict[str, str] = {}
        self._epoch = 0
        self._confirmation: LevelCheck | None = None
        self._checks = 0
        self._clears = 0
        self._hold_active = False
        self._hold_episode = 0
        self._release: dict | None = None
        self._lock = threading.RLock()

    @property
    def level_mm(self) -> float:
        """当前采信的液面高度（双路仲裁结论）。"""

        with self._lock:
            return self._arbitrate_locked().value_mm

    @property
    def epoch(self) -> int:
        """确认凭据的代次；清炉或出料后加一。"""

        with self._lock:
            return self._epoch

    def observe(self, level_mm: float) -> float:
        """更新真实液面：两路测量按各自的测量偏移跟随。

        偏移来自单路注入（结料、漂移的演练）；没有注入时两路读数相同。
        """

        value = float(level_mm)
        with self._lock:
            self._process_mm = value
            for channel in self._channels.values():
                channel.write(value + channel.offset_mm, self._clock.now())
            self._refresh_locked()
        self._bus.publish(
            Event.of(
                "furnace",
                "furnace",
                "level_observed",
                channel="both",
                level_mm=round(value, 3),
            )
        )
        return self.level_mm

    def observe_channel(self, channel: str, level_mm: float) -> float:
        """单路变送器来数：写入该通道读数，等效于修正该通道的测量偏移。"""

        value = float(level_mm)
        with self._lock:
            state = self._channel(channel)
            state.offset_mm = value - self._process_mm
            state.write(value, self._clock.now())
            self._refresh_locked()
        self._bus.publish(
            Event.of(
                "furnace",
                "furnace",
                "level_observed",
                channel=channel,
                level_mm=round(value, 3),
            )
        )
        return value

    def set_channel_fault(self, channel: str, faulty: bool, reason: str = "") -> None:
        """置位／复位某一路的变送器自诊断。"""

        with self._lock:
            state = self._channel(channel)
            if faulty:
                state.fault = reason or "自诊断置位"
            else:
                state.fault = None
            self._refresh_locked()

    def drift(self, delta_mm: float) -> float:
        """按投料／出料量调整真实液面，返回新的采信液面高度。"""

        with self._lock:
            process = self._process_mm
        return self.observe(process + float(delta_mm))

    def check(self) -> LevelCheck:
        """现场复核液面：按当前双路仲裁结论重新判定，并把结果记为最新确认。

        两路打架且未放行时一律判不通过——不是液面一定越带，而是不知道该信
        哪一路，必须等人到现场核实。
        """

        with self._lock:
            self._refresh_locked()
            arb = self._arbitrate_locked()
            released = False
            note = None
            if arb.status in ("disagree", "no_signal"):
                release = self._release
                if release is not None and release["episode"] == self._hold_episode:
                    trust = release["trust"]
                    basis = (
                        self._channels[trust].value_mm if trust is not None else arb.value_mm
                    )
                    ok = self._limits.level_ok(basis)
                    level_mm = basis
                    hold = False
                    released = True
                    note = release["note"]
                else:
                    ok = False
                    level_mm = arb.value_mm
                    hold = True
            else:
                ok = self._limits.level_ok(arb.value_mm)
                level_mm = arb.value_mm
                hold = False
            result = LevelCheck(
                level_mm=level_mm,
                target_mm=self._limits.level_target_mm,
                band_mm=self._limits.level_band_mm,
                ok=ok,
                high=self._limits.level_high(level_mm),
                epoch=self._epoch,
                checked_at=self._clock.now(),
                status=arb.status,
                channel=arb.trusted,
                untrusted="+".join(arb.untrusted) if arb.untrusted else None,
                diff_mm=arb.diff_mm,
                hold=hold,
                released=released,
                note=note,
            )
            self._confirmation = result
            self._checks += 1
        self._bus.publish(
            Event.of(
                "furnace",
                "furnace",
                "level_checked",
                subject="glass_level",
                severity="info" if result.ok else "warning",
                level_mm=round(result.level_mm, 3),
                epoch=result.epoch,
                ok=result.ok,
                status=result.status,
                untrusted=result.untrusted,
                hold=result.hold,
                released=result.released,
            )
        )
        return result

    def release_hold(self, note: str, trust: str | None = None) -> dict:
        """现场核实后放行当前这轮分歧持留。

        ``trust`` 给出经现场核实后采信的通道；缺省按两路保守高值判定。
        放行只在当前分歧轮次内有效，双路重新一致后自动作废。
        """

        with self._lock:
            if not self._hold_active:
                raise InterlockError("当前没有待放行的液面持留", component="furnace")
            if trust is not None and trust not in self._channels:
                raise SequenceError(f"未知的液面测量通道: {trust}", component="furnace")
            self._release = {
                "episode": self._hold_episode,
                "trust": trust,
                "note": note,
                "at": self._clock.now(),
            }
            episode = self._hold_episode
        self._bus.publish(
            Event.of(
                "furnace",
                "furnace",
                "level_hold_released",
                subject="glass_level",
                severity="notice",
                episode=episode,
                trust=trust,
                note=note,
            )
        )
        return {
            "episode": episode,
            "trust": trust,
            "note": note,
            "level": self.snapshot(),
        }

    def last_confirmation(self) -> LevelCheck | None:
        """返回最近一次确认结果（仅供展示与审计，不代替现场复核）。"""

        with self._lock:
            return self._confirmation

    def clear(self) -> int:
        """出料或清炉后作废确认凭据与未决放行，返回作废后的代次。"""

        with self._lock:
            self._confirmation = None
            self._release = None
            self._epoch += 1
            self._clears += 1
            epoch = self._epoch
        self._bus.publish(
            Event.of(
                "furnace",
                "furnace",
                "level_cleared",
                subject="glass_level",
                severity="notice",
                epoch=epoch,
            )
        )
        return epoch

    def counts(self) -> dict:
        with self._lock:
            return {"checks": self._checks, "clears": self._clears}

    def snapshot(self) -> dict:
        confirmation = self.last_confirmation()
        counts = self.counts()
        with self._lock:
            arb = self._arbitrate_locked()
            release_valid = (
                self._release is not None
                and self._release["episode"] == self._hold_episode
            )
            if arb.status in ("disagree", "no_signal"):
                if release_valid:
                    trust = self._release["trust"]
                    basis = (
                        self._channels[trust].value_mm if trust is not None else arb.value_mm
                    )
                    level_ok = self._limits.level_ok(basis)
                else:
                    basis = arb.value_mm
                    level_ok = False
            else:
                basis = arb.value_mm
                level_ok = self._limits.level_ok(basis)
            return {
                "level_mm": round(basis, 3),
                "level_ok": level_ok,
                "level_high": self._limits.level_high(basis),
                "target_mm": self._limits.level_target_mm,
                "band_mm": self._limits.level_band_mm,
                "agree_mm": self._limits.level_agree_mm,
                "status": arb.status,
                "trusted": arb.trusted,
                "untrusted": list(arb.untrusted),
                "diff_mm": round(arb.diff_mm, 3),
                "channels": {
                    name: {
                        "principle": CHANNEL_PRINCIPLE[name],
                        "value_mm": round(self._channels[name].value_mm, 3),
                        "offset_mm": round(self._channels[name].offset_mm, 3),
                        "trusted": name not in arb.untrusted,
                        "reason": arb.reasons.get(name, ""),
                        "fault": self._channels[name].fault is not None,
                    }
                    for name in CHANNEL_PRINCIPLE
                },
                "hold": {
                    "active": self._hold_active,
                    "episode": self._hold_episode,
                    "released": release_valid,
                    "release": dict(self._release) if release_valid else None,
                },
                "epoch": self._epoch,
                "checks": counts["checks"],
                "clears": counts["clears"],
                "last_confirmation": confirmation.as_dict() if confirmation else None,
            }

    def _channel(self, name: str) -> _ChannelState:
        try:
            return self._channels[name]
        except KeyError:
            raise SequenceError(f"未知的液面测量通道: {name}", component="furnace") from None

    def _channel_health(self, state: _ChannelState) -> str:
        """返回通道不可信的原因；空串表示可信。"""

        if state.fault is not None:
            return f"变送器自诊断置位：{state.fault}"
        value = state.value_mm
        if (
            not math.isfinite(value)
            or value < 0.0
            or value > self._limits.level_plausible_max_mm
        ):
            return "读数越出物理可信范围"
        if state.same_count >= self._limits.level_frozen_count:
            return "读数连续冻结，疑似结料卡死"
        return ""

    def _arbitrate_locked(self) -> Arbitration:
        reasons = {
            name: self._channel_health(self._channels[name]) for name in CHANNEL_PRINCIPLE
        }
        untrusted = tuple(name for name, reason in reasons.items() if reason)
        probe = self._channels["probe"].value_mm
        radar = self._channels["radar"].value_mm
        diff = abs(probe - radar)
        if len(untrusted) == 2:
            return Arbitration("no_signal", max(probe, radar), "none", untrusted, diff, reasons)
        if len(untrusted) == 1:
            trusted = "radar" if untrusted[0] == "probe" else "probe"
            return Arbitration(
                "degraded",
                self._channels[trusted].value_mm,
                trusted,
                untrusted,
                diff,
                reasons,
            )
        if diff <= self._limits.level_agree_mm:
            return Arbitration("agreed", (probe + radar) / 2.0, "both", untrusted, diff, reasons)
        return Arbitration("disagree", max(probe, radar), "none", untrusted, diff, reasons)

    def _refresh_locked(self) -> None:
        """刷新通道可信边沿与持留状态，并在状态迁移时发事件。"""

        arb = self._arbitrate_locked()
        for name, reason in arb.reasons.items():
            was = self._rejected.get(name)
            if reason and was is None:
                self._rejected[name] = reason
                self._bus.publish(
                    Event.of(
                        "furnace",
                        "furnace",
                        "level_channel_rejected",
                        subject="glass_level",
                        severity="warning",
                        channel=name,
                        principle=CHANNEL_PRINCIPLE[name],
                        reason=reason,
                    )
                )
            elif not reason and was is not None:
                del self._rejected[name]
                self._bus.publish(
                    Event.of(
                        "furnace",
                        "furnace",
                        "level_channel_recovered",
                        subject="glass_level",
                        severity="notice",
                        channel=name,
                        principle=CHANNEL_PRINCIPLE[name],
                    )
                )
        if arb.status in ("disagree", "no_signal"):
            if not self._hold_active:
                self._hold_active = True
                self._hold_episode += 1
                self._release = None
                self._bus.publish(
                    Event.of(
                        "furnace",
                        "furnace",
                        "level_hold_raised",
                        subject="glass_level",
                        severity="warning",
                        episode=self._hold_episode,
                        status=arb.status,
                        diff_mm=round(arb.diff_mm, 3),
                        probe_mm=round(self._channels["probe"].value_mm, 3),
                        radar_mm=round(self._channels["radar"].value_mm, 3),
                    )
                )
        elif self._hold_active:
            self._hold_active = False
            self._release = None
            self._bus.publish(
                Event.of(
                    "furnace",
                    "furnace",
                    "level_hold_cleared",
                    subject="glass_level",
                    severity="notice",
                    episode=self._hold_episode,
                    status=arb.status,
                )
            )
