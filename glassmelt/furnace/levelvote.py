"""双路液面互校器。

投料门控的液面结论由两路不同原理的测量互校后给出：一路是接触式探针
（会受液面波动漂移、结料影响），另一路是非接触雷达（不受结料影响，
失效形态与探针不同）。互校器逐拍给每一路打可信标记，再按下面四种情形
表决：

* ``agree``      两路都健康且读数一致——采信均值，正常投料；
* ``degraded``   只有一路健康——采信剩下那一路，并标明哪一路不可信；
* ``disagreed``  两路都健康但读数打架——不给自动结论，等人到现场确认；
* ``blind``      两路都不可信——盲态，禁止投料，现场放行也不允许。

单路跳变只在"这一路动、另一路没跟着动"时判坏：两路在同一拍同步跳同样
的幅度是真实液面变化（出料、塌料），不能因为一次共同跳变白停料。探针
结料（读数长期挂住不动）靠冻结判据识别：一路在时间窗内基本不动而另一
路健康且已经偏离，就把挂住的这一路摘掉。
"""

from __future__ import annotations

import threading
from collections import deque
from dataclasses import dataclass
from typing import Deque

from ..clock import Clock
from ..config import FurnaceLimits
from ..eventbus import Event, EventBus

#: 接触式探针：原有那一路测量。
CHANNEL_PROBE = "probe"
#: 非接触雷达：不同原理的第二路测量。
CHANNEL_RADAR = "radar"

CHANNEL_NAMES: dict[str, str] = {
    CHANNEL_PROBE: "接触式探针",
    CHANNEL_RADAR: "非接触雷达",
}
CHANNELS: tuple[str, str] = (CHANNEL_PROBE, CHANNEL_RADAR)

#: 两路一致，正常采信。
MODE_AGREE = "agree"
#: 单路降级：一路坏，用剩下那路。
MODE_DEGRADED = "degraded"
#: 两路打架：停自动投料，等现场确认。
MODE_DISAGREED = "disagreed"
#: 双盲：两路都不可信，禁止投料。
MODE_BLIND = "blind"

MODE_NAMES: dict[str, str] = {
    MODE_AGREE: "双路一致",
    MODE_DEGRADED: "单路降级",
    MODE_DISAGREED: "双路分歧",
    MODE_BLIND: "双路全盲",
}

#: 故障码到现场可读说明。
FAULT_NAMES: dict[str, str] = {
    "spike": "单点跳变",
    "stuck": "读数冻结（疑似结料）",
    "coating": "挂料（已挂牌）",
    "range": "超量程",
}

_HISTORY_LIMIT = 512


@dataclass(frozen=True)
class ChannelState:
    """一路液面测量的当前可信状态。"""

    channel: str
    name: str
    raw_mm: float | None
    healthy: bool
    seen: bool
    stale: bool
    fault: str | None
    updated_at: float | None

    def as_dict(self) -> dict:
        return {
            "channel": self.channel,
            "name": self.name,
            "raw_mm": round(self.raw_mm, 3) if self.raw_mm is not None else None,
            "healthy": self.healthy,
            "seen": self.seen,
            "stale": self.stale,
            "fault": self.fault,
            "fault_name": FAULT_NAMES.get(self.fault or "", self.fault),
            "updated_at": round(self.updated_at, 3) if self.updated_at is not None else None,
        }


@dataclass(frozen=True)
class VoteResult:
    """一次双路互校的表决结论。"""

    mode: str
    mode_name: str
    adopted_mm: float | None
    divergence_mm: float | None
    trusted_channel: str | None
    suspect_channel: str | None
    reason: str
    channels: dict[str, ChannelState]

    @property
    def trusted(self) -> bool:
        """本拍结论是否可用于投料门控（双路一致或单路降级）。"""

        return self.mode in (MODE_AGREE, MODE_DEGRADED)

    def as_dict(self) -> dict:
        return {
            "mode": self.mode,
            "mode_name": self.mode_name,
            "trusted": self.trusted,
            "adopted_mm": round(self.adopted_mm, 3) if self.adopted_mm is not None else None,
            "divergence_mm": (
                round(self.divergence_mm, 3) if self.divergence_mm is not None else None
            ),
            "trusted_channel": self.trusted_channel,
            "suspect_channel": self.suspect_channel,
            "reason": self.reason,
            "channels": {
                channel: state.as_dict() for channel, state in self.channels.items()
            },
        }


class LevelChannel:
    """一路液面测量通道：保留近期样本与已锁定的故障标记。"""

    def __init__(self, channel: str) -> None:
        self.channel = channel
        self.name = CHANNEL_NAMES[channel]
        self._history: Deque[tuple[float, float]] = deque(maxlen=_HISTORY_LIMIT)
        self._raw_mm: float | None = None
        self._updated_at: float | None = None
        self._fault: str | None = None
        self._prev_mm: float | None = None

    @property
    def raw_mm(self) -> float | None:
        return self._raw_mm

    @property
    def fault(self) -> str | None:
        return self._fault

    @property
    def seen(self) -> bool:
        return self._raw_mm is not None

    @property
    def updated_at(self) -> float | None:
        return self._updated_at

    def history(self) -> list[tuple[float, float]]:
        return list(self._history)

    def moved_from_previous(self) -> bool:
        """本拍样本是否相对上一拍发生了变化。"""

        return self._prev_mm is not None and self._raw_mm != self._prev_mm

    @property
    def jump_mm(self) -> float | None:
        """本拍相对上一拍的跳变量（无上一拍时为 None）。"""

        if self._raw_mm is None or self._prev_mm is None:
            return None
        return self._raw_mm - self._prev_mm

    def update(self, now: float, value: float, limits: FurnaceLimits) -> str | None:
        """收一个新样本；超量程立即锁死，返回本次新锁定的故障码。"""

        value = float(value)
        self._prev_mm = self._raw_mm
        self._raw_mm = value
        self._updated_at = now
        self._history.append((now, value))
        if self._fault is None and not (
            limits.level_min_mm <= value <= limits.level_max_mm
        ):
            self._fault = "range"
        return self._fault if self._fault == "range" else None

    def latch(self, fault: str) -> str | None:
        """挂上一个需维护复位的故障标记（跳变、冻结、结料）。"""

        if self._fault is None:
            self._fault = fault
            return fault
        return None

    def mark_coating(self) -> str | None:
        """现场挂牌：探头结料挂料，本路退出表决直到维护复位。"""

        return self.latch("coating")

    def reset(self) -> str | None:
        """维护复位：清故障标记与旧样本，从下一拍重新建立可信基线。"""

        cleared = self._fault
        self._fault = None
        self._history.clear()
        self._raw_mm = None
        self._updated_at = None
        self._prev_mm = None
        return cleared

    def health_at(self, now: float, stale_after_s: float | None = None) -> ChannelState:
        """按当前时刻给出健康状态；超时未刷新算失联（瞬态，不锁定）。"""

        stale = (
            self._updated_at is not None
            and stale_after_s is not None
            and now - self._updated_at > stale_after_s
        )
        healthy = self._fault is None and self._raw_mm is not None and not stale
        return ChannelState(
            channel=self.channel,
            name=self.name,
            raw_mm=self._raw_mm,
            healthy=healthy,
            seen=self._raw_mm is not None,
            stale=bool(stale),
            fault=self._fault,
            updated_at=self._updated_at,
        )


class LevelVoter:
    """两路液面测量的健康判定与互校表决。"""

    def __init__(self, clock: Clock, bus: EventBus, limits: FurnaceLimits) -> None:
        self._clock = clock
        self._bus = bus
        self._limits = limits
        self._probe = LevelChannel(CHANNEL_PROBE)
        self._radar = LevelChannel(CHANNEL_RADAR)
        self._lock = threading.RLock()
        self._faults: dict[str, str | None] = {
            CHANNEL_PROBE: None,
            CHANNEL_RADAR: None,
        }
        self._vote = self._initial_vote()
        # 模式迁移回调（例如分歧解除时让未消费的现场放行凭据作废）。
        self.on_mode_change: object = None
        # 本轮（两次 evaluate 之间）真正收到新样本的通道。
        self._fresh: set[str] = set()

    @property
    def vote(self) -> VoteResult:
        with self._lock:
            return self._vote

    def channel(self, channel: str) -> LevelChannel:
        return self._channels()[channel]

    def update(self, channel: str, value: float) -> VoteResult:
        """某一路送来一个新样本，随后重新互校表决。"""

        with self._lock:
            self.ingest(channel, value)
            return self.evaluate()

    def ingest(self, channel: str, value: float) -> None:
        """只收一路新样本（不立即表决），供两路同拍送样后统一互校。"""

        with self._lock:
            now = self._clock.now()
            newly = self.channel(channel).update(now, value, self._limits)
            self._fresh.add(channel)
            self._bus.publish(
                Event.of(
                    "furnace",
                    "furnace",
                    "level_channel_observed",
                    subject="glass_level",
                    severity="info",
                    channel=channel,
                    channel_name=CHANNEL_NAMES[channel],
                    level_mm=round(float(value), 3),
                )
            )
            if newly:
                self._publish_fault(channel, newly)

    def evaluate_pair(self) -> VoteResult:
        """两路都已送完本拍样本后统一互校表决。"""

        with self._lock:
            return self.evaluate()

    def mark_coating(self, channel: str) -> VoteResult:
        """把某一路挂牌为结料挂料，随后重新表决。"""

        with self._lock:
            newly = self.channel(channel).mark_coating()
            if newly:
                self._publish_fault(channel, newly)
            return self.evaluate()

    def reset_channel(self, channel: str, reason: str) -> dict:
        """维护后让某一路重新参与表决。"""

        with self._lock:
            cleared = self.channel(channel).reset()
            # 复位后允许该通道同类故障再次发生时重新上报告警事件。
            self._faults[channel] = None
            self._fresh.discard(channel)
            self._bus.publish(
                Event.of(
                    "furnace",
                    "furnace",
                    "level_channel_reset",
                    subject="glass_level",
                    severity="notice",
                    channel=channel,
                    channel_name=CHANNEL_NAMES[channel],
                    cleared_fault=cleared,
                    reason=reason,
                )
            )
            vote = self.evaluate()
            return {
                "channel": channel,
                "channel_name": CHANNEL_NAMES[channel],
                "cleared_fault": cleared,
                "reason": reason,
                "vote": vote.as_dict(),
            }

    def evaluate(self) -> VoteResult:
        """按两路最新样本与故障标记重新判定并发布状态迁移事件。"""

        with self._lock:
            now = self._clock.now()
            self._detect_spike(now)
            self._detect_stuck(now)
            self._fresh.clear()

            states = {
                CHANNEL_PROBE: self._probe.health_at(now, self._limits.level_stale_s),
                CHANNEL_RADAR: self._radar.health_at(now, self._limits.level_stale_s),
            }
            vote = self._ballot(states)

            previous_mode = self._vote.mode
            if vote.mode != previous_mode:
                self._publish_mode(vote, previous_mode)
                hook = self.on_mode_change
                if callable(hook):
                    hook(vote, previous_mode)
            self._vote = vote
            return vote

    def snapshot(self) -> dict:
        with self._lock:
            return self._vote.as_dict()

    def _channels(self) -> dict[str, LevelChannel]:
        return {CHANNEL_PROBE: self._probe, CHANNEL_RADAR: self._radar}

    def _detect_spike(self, now: float) -> None:
        """只怀疑"这一拍只有它在动且动作过大"的那一路。

        两路本拍同步变化（出料、塌料引起的真实液面变化）不判坏；一路随
        出料缓慢漂移也不判坏——那种"别人都在动它却不动"的情形交给冻结
        判据识别。
        """

        movers: list[LevelChannel] = []
        for channel in (self._probe, self._radar):
            if (
                channel.channel in self._fresh
                and channel.fault is None
                and channel.raw_mm is not None
                and channel.moved_from_previous()
            ):
                movers.append(channel)
        if len(movers) != 1:
            return
        mover = movers[0]
        other = self._radar if mover is self._probe else self._probe
        other_state = other.health_at(now, self._limits.level_stale_s)
        jump = mover.jump_mm
        if not other_state.healthy or other.raw_mm is None or jump is None:
            return
        if abs(jump) > self._limits.level_spike_mm:
            fault = mover.latch("spike")
            if fault:
                self._publish_fault(mover.channel, fault)

    def _detect_stuck(self, now: float) -> None:
        """一路在时间窗内几乎不动、另一路健康且已明显偏离：挂住的一路判坏。"""

        for flat, other in ((self._probe, self._radar), (self._radar, self._probe)):
            if flat.fault is not None or flat.raw_mm is None:
                continue
            cutoff = now - self._limits.level_stuck_span_s
            window = [(stamp, value) for stamp, value in flat.history() if stamp >= cutoff]
            if not window or window[0][0] > cutoff:
                # 样本还没覆盖整个冻结判据窗，证据不足。
                continue
            spread = max(value for _, value in window) - min(value for _, value in window)
            if spread > self._limits.level_stuck_deadband_mm:
                continue
            other_state = other.health_at(now, self._limits.level_stale_s)
            if not other_state.healthy or other.raw_mm is None:
                continue
            if abs(other.raw_mm - flat.raw_mm) > self._limits.level_vote_tolerance_mm:
                fault = flat.latch("stuck")
                if fault:
                    self._publish_fault(flat.channel, fault)

    def _ballot(self, states: dict[str, ChannelState]) -> VoteResult:
        probe = states[CHANNEL_PROBE]
        radar = states[CHANNEL_RADAR]
        divergence = (
            abs(probe.raw_mm - radar.raw_mm)
            if probe.raw_mm is not None and radar.raw_mm is not None
            else None
        )

        if probe.healthy and radar.healthy:
            same_verdict = self._limits.level_ok(
                probe.raw_mm
            ) == self._limits.level_ok(radar.raw_mm) and self._limits.level_high(
                probe.raw_mm
            ) == self._limits.level_high(radar.raw_mm)
            if divergence is not None and divergence <= self._limits.level_vote_tolerance_mm:
                if same_verdict:
                    return self._result(
                        MODE_AGREE,
                        (probe.raw_mm + radar.raw_mm) / 2.0,
                        divergence,
                        reason=(
                            f"探针 {probe.raw_mm:.1f} mm 与雷达 {radar.raw_mm:.1f} mm 一致"
                        ),
                        states=states,
                    )
                return self._result(
                    MODE_DISAGREED,
                    None,
                    divergence,
                    reason="两路读数接近但高低判定相反，需现场核认",
                    states=states,
                )
            return self._result(
                MODE_DISAGREED,
                None,
                divergence,
                reason=(
                    f"两路读数相差 {divergence:.1f} mm（超过互校公差 "
                    f"{self._limits.level_vote_tolerance_mm:.1f} mm）"
                ),
                states=states,
            )

        if probe.healthy != radar.healthy:
            healthy_channel = CHANNEL_PROBE if probe.healthy else CHANNEL_RADAR
            bad_channel = CHANNEL_RADAR if probe.healthy else CHANNEL_PROBE
            bad = states[bad_channel]
            adopted = states[healthy_channel].raw_mm
            detail = FAULT_NAMES.get(bad.fault or "", "失联") if bad.fault else "数据失联"
            return self._result(
                MODE_DEGRADED,
                adopted,
                divergence,
                trusted=healthy_channel,
                suspect=bad_channel,
                reason=f"{CHANNEL_NAMES[bad_channel]}{detail}，本拍采信{CHANNEL_NAMES[healthy_channel]}",
                states=states,
            )

        reasons = []
        for state in (probe, radar):
            if not state.seen:
                reasons.append(f"{state.name}未送样")
            elif state.fault:
                reasons.append(f"{state.name}{FAULT_NAMES.get(state.fault, state.fault)}")
            elif state.stale:
                reasons.append(f"{state.name}数据失联")
        return self._result(
            MODE_BLIND,
            None,
            divergence,
            reason="两路液面均不可信：" + "、".join(reasons),
            states=states,
        )

    def _result(
        self,
        mode: str,
        adopted_mm: float | None,
        divergence_mm: float | None,
        reason: str,
        states: dict[str, ChannelState],
        trusted: str | None = None,
        suspect: str | None = None,
    ) -> VoteResult:
        return VoteResult(
            mode=mode,
            mode_name=MODE_NAMES[mode],
            adopted_mm=adopted_mm,
            divergence_mm=divergence_mm,
            trusted_channel=trusted,
            suspect_channel=suspect,
            reason=reason,
            channels=dict(states),
        )

    def _initial_vote(self) -> VoteResult:
        states = {
            channel: self.channel(channel).health_at(self._clock.now())
            for channel in CHANNELS
        }
        return self._result(
            MODE_BLIND, None, None, "两路液面均尚未送样", states
        )

    def _publish_fault(self, channel: str, fault: str) -> None:
        previous = self._faults.get(channel)
        if previous == fault:
            return
        self._faults[channel] = fault
        self._bus.publish(
            Event.of(
                "furnace",
                "furnace",
                "level_channel_fault",
                subject="glass_level",
                severity="warning",
                channel=channel,
                channel_name=CHANNEL_NAMES[channel],
                fault=fault,
                fault_name=FAULT_NAMES.get(fault, fault),
            )
        )

    def _publish_mode(self, vote: VoteResult, previous_mode: str) -> None:
        severity = {
            MODE_AGREE: "notice",
            MODE_DEGRADED: "warning",
            MODE_DISAGREED: "alarm",
            MODE_BLIND: "alarm",
        }[vote.mode]
        self._bus.publish(
            Event.of(
                "furnace",
                "furnace",
                "level_vote_mode",
                subject="glass_level",
                severity=severity,
                mode=vote.mode,
                mode_name=vote.mode_name,
                previous_mode=previous_mode,
                adopted_mm=(
                    round(vote.adopted_mm, 3) if vote.adopted_mm is not None else None
                ),
                divergence_mm=(
                    round(vote.divergence_mm, 3)
                    if vote.divergence_mm is not None
                    else None
                ),
                trusted_channel=vote.trusted_channel,
                suspect_channel=vote.suspect_channel,
                reason=vote.reason,
            )
        )
