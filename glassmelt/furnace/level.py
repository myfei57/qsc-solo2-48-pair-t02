"""玻璃液面监测、双路互校与确认。

投料前必须先确认窑内液面：液面过低才能补料，液面一旦被压到允许带以上，
继续投料会把生料堆过熔化线。因此"确认"不能是一份长期有效的凭据——每次
投料动作都要按当前液面重新判一遍。出料或清炉后确认凭据必须作废，否则
恢复投料会拿着作废的确认继续加料。

液面结论不来自单支探头，而由两路不同原理的测量（接触式探针＋非接触雷达）
经 :class:`~glassmelt.furnace.levelvote.LevelVoter` 互校后给出：两路一致才
采信；单路坏用剩下那路并标明不可信的一路；两路打架不给自动结论，必须由
人到现场确认后换一张一次性的"现场放行凭据"，且放行凭据只对紧随其后的一
次投料有效；两路全盲时任何放行都不接受。
"""

from __future__ import annotations

import threading
import uuid
from dataclasses import dataclass

from ..clock import Clock
from ..config import FurnaceLimits
from ..errors import InterlockError
from ..eventbus import Event, EventBus
from .levelvote import (
    CHANNELS,
    CHANNEL_NAMES,
    MODE_BLIND,
    MODE_DISAGREED,
    MODE_NAMES,
    LevelChannel,
    LevelVoter,
)


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
    mode: str = MODE_BLIND
    mode_name: str = MODE_NAMES[MODE_BLIND]
    divergence_mm: float | None = None
    trusted_channel: str | None = None
    suspect_channel: str | None = None
    manual_release: bool = False
    released_by: str | None = None

    def as_dict(self) -> dict:
        return {
            "level_mm": round(self.level_mm, 3),
            "target_mm": self.target_mm,
            "band_mm": self.band_mm,
            "ok": self.ok,
            "high": self.high,
            "epoch": self.epoch,
            "checked_at": round(self.checked_at, 3),
            "mode": self.mode,
            "mode_name": self.mode_name,
            "divergence_mm": (
                round(self.divergence_mm, 3) if self.divergence_mm is not None else None
            ),
            "trusted_channel": self.trusted_channel,
            "suspect_channel": self.suspect_channel,
            "manual_release": self.manual_release,
            "released_by": self.released_by,
        }


@dataclass(frozen=True)
class ManualRelease:
    """现场确认后换发的一次性投料放行凭据。"""

    release_id: str
    issued_at: float
    expires_at: float
    issued_by: str
    reason: str
    observed_mm: float

    def as_dict(self) -> dict:
        return {
            "release_id": self.release_id,
            "issued_at": round(self.issued_at, 3),
            "expires_at": round(self.expires_at, 3),
            "issued_by": self.issued_by,
            "reason": self.reason,
            "observed_mm": round(self.observed_mm, 3),
        }


class GlassLevel:
    """窑内双路液面测量、互校表决与确认缓存。"""

    def __init__(self, clock: Clock, bus: EventBus, limits: FurnaceLimits) -> None:
        self._clock = clock
        self._bus = bus
        self._limits = limits
        self._voter = LevelVoter(clock, bus, limits)
        self._voter.on_mode_change = self._on_vote_mode_change
        self._level_mm = limits.level_target_mm
        self._epoch = 0
        self._confirmation: LevelCheck | None = None
        self._release: ManualRelease | None = None
        self._checks = 0
        self._clears = 0
        self._releases = 0
        self._lock = threading.RLock()

    @property
    def level_mm(self) -> float:
        """当前采信液面；尚无两路结论时维持初始目标值。"""

        with self._lock:
            return self._level_mm

    @property
    def epoch(self) -> int:
        """确认凭据的代次；清炉或出料后加一。"""

        with self._lock:
            return self._epoch

    def channel(self, channel: str) -> LevelChannel:
        return self._voter.channel(channel)

    def channels(self) -> tuple[str, ...]:
        return CHANNELS

    def observe(self, level_mm: float, channel: str | None = None) -> float:
        """更新液面测量。

        不指定通道时两路同时喂同一个值（仿真／标定用，两路必然一致）；
        指定 ``probe`` 或 ``radar`` 时只更新对应那一路。
        """

        with self._lock:
            if channel is not None:
                targets = (channel,)
            else:
                targets = CHANNELS
            for name in targets:
                self._voter.ingest(name, float(level_mm))
            vote = (
                self._voter.evaluate_pair() if channel is None
                else self._voter.evaluate()
            )
            value = self._adopt(vote)
        return value

    def observe_channel(self, channel: str, level_mm: float) -> float:
        """只更新指定一路测量并重新互校。"""

        return self.observe(level_mm, channel=channel)

    def drift(self, delta_mm: float, channel: str | None = None) -> float:
        """按投料／出料量调整液面，返回调整后采信值。

        默认两路同步漂移；指定通道时只推一路（用于复现单路漂移／结料）。
        """

        with self._lock:
            if channel is not None:
                current = self.channel(channel).raw_mm
                if current is None:
                    current = self._level_mm
                self._voter.ingest(channel, current + float(delta_mm))
                vote = self._voter.evaluate()
                return self._adopt(vote)
            values = {
                name: (self.channel(name).raw_mm or self._level_mm) + float(delta_mm)
                for name in CHANNELS
            }
            for name in CHANNELS:
                self._voter.ingest(name, values[name])
            return self._adopt(self._voter.evaluate_pair())

    def mark_coating(self, channel: str, reason: str = "现场挂牌结料") -> dict:
        """把某一路挂牌为结料挂料（探头结料的现场处置入口）。"""

        with self._lock:
            vote = self._voter.mark_coating(channel)
            self._adopt(vote)
            self._bus.publish(
                Event.of(
                    "furnace",
                    "furnace",
                    "level_channel_coating",
                    subject="glass_level",
                    severity="warning",
                    channel=channel,
                    channel_name=CHANNEL_NAMES[channel],
                    reason=reason,
                )
            )
            return vote.as_dict()

    def reset_channel(self, channel: str, reason: str = "维护后恢复") -> dict:
        """维护复位某一路，让它重新参与互校。"""

        with self._lock:
            result = self._voter.reset_channel(channel, reason)
            self._adopt(self._voter.vote)
            return result

    def release_for_site_check(
        self, issued_by: str, reason: str, observed_mm: float
    ) -> ManualRelease:
        """两路打架时由人到现场确认，换发一张一次性投料放行凭据。

        必须登记操作人、现场确认说明和现场读到的实际液位；现场观测液位
        若已在允许带以上，即便有人签发放行也不允许投料，避免顶着熔化线
        加料。
        """

        operator = (issued_by or "").strip()
        detail = (reason or "").strip()
        if not operator:
            raise InterlockError(
                "现场放行必须登记操作人（谁到现场看的）", component="furnace"
            )
        if not detail:
            raise InterlockError(
                "现场放行必须填写现场确认说明", component="furnace"
            )
        try:
            observed = float(observed_mm)
        except (TypeError, ValueError):
            raise InterlockError(
                "现场放行必须登记现场实测液位（毫米）", component="furnace"
            )
        with self._lock:
            vote = self._voter.evaluate()
            if vote.mode != MODE_DISAGREED:
                raise InterlockError(
                    f"当前液面状态为{vote.mode_name}，不是双路分歧，"
                    "不允许签发现场放行凭据",
                    component="furnace",
                )
            if not self._limits.level_ok(observed):
                raise InterlockError(
                    f"现场实测液位 {observed:.1f} mm 已在允许带"
                    f"（目标 {self._limits.level_target_mm:.1f} ± "
                    f"{self._limits.level_band_mm:.1f} mm）以外，禁止投料放行",
                    component="furnace",
                )
            now = self._clock.now()
            release = ManualRelease(
                release_id=str(uuid.uuid4())[:8],
                issued_at=now,
                expires_at=now + self._limits.level_release_window_s,
                issued_by=operator,
                reason=detail,
                observed_mm=observed,
            )
            self._release = release
            self._releases += 1
            self._bus.publish(
                Event.of(
                    "furnace",
                    "furnace",
                    "level_disagreement_released",
                    subject="glass_level",
                    severity="notice",
                    release_id=release.release_id,
                    issued_by=operator,
                    reason=detail,
                    observed_mm=round(observed, 3),
                    expires_at=round(release.expires_at, 3),
                    divergence_mm=(
                        round(vote.divergence_mm, 3)
                        if vote.divergence_mm is not None
                        else None
                    ),
                )
            )
            return release

    def check(self) -> LevelCheck:
        """现场复核液面：始终按当前互校结论重新判定，并记为最新确认。"""

        with self._lock:
            vote = self._voter.evaluate()
            self._adopt(vote)

            release = self._consume_release_if_allowed(vote)
            if vote.mode == MODE_DISAGREED and release is None:
                result = self._build_check(vote, None, ok=False)
                self._record_check(result)
                raise InterlockError(
                    f"两路液面打架（{vote.reason}），已停自动投料；"
                    "请人到现场核认，确认后签发一次性现场放行再投料",
                    component="batch",
                )
            if vote.mode == MODE_BLIND:
                result = self._build_check(vote, None, ok=False)
                self._record_check(result)
                raise InterlockError(
                    f"两路液面均不可信（{vote.reason}），禁止投料；"
                    "请先恢复至少一路测量",
                    component="batch",
                )

            result = self._build_check(vote, release)
            self._record_check(result)
            return result
    def last_confirmation(self) -> LevelCheck | None:
        """返回最近一次确认结果（仅供展示与审计，不代替现场复核）。"""

        with self._lock:
            return self._confirmation

    def pending_release(self) -> ManualRelease | None:
        """尚未消费的现场放行凭据（仅供展示与审计）。"""

        with self._lock:
            return self._release

    def clear(self) -> int:
        """出料或清炉后作废确认凭据与未消费的放行，返回作废后的代次。"""

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
            return {"checks": self._checks, "clears": self._clears, "releases": self._releases}

    def snapshot(self) -> dict:
        with self._lock:
            confirmation = self._confirmation
            release = self._release
            vote = self._voter.snapshot()
            adopted = vote["adopted_mm"] if vote["adopted_mm"] is not None else self._level_mm
            counts = self.counts()
            return {
                "level_mm": round(adopted, 3),
                "level_ok": self._limits.level_ok(adopted),
                "level_high": self._limits.level_high(adopted),
                "target_mm": self._limits.level_target_mm,
                "band_mm": self._limits.level_band_mm,
                "epoch": self._epoch,
                "checks": counts["checks"],
                "clears": counts["clears"],
                "releases": counts["releases"],
                "vote": vote,
                "channels": vote["channels"],
                "pending_release": release.as_dict() if release else None,
                "last_confirmation": confirmation.as_dict() if confirmation else None,
            }

    def _build_check(
        self, vote, release: ManualRelease | None, ok: bool | None = None
    ) -> LevelCheck:
        if release is not None:
            # 现场放行：以人在现场登记的观测液位做门控，两路读数只留作分歧佐证。
            adopted = release.observed_mm
        elif vote.adopted_mm is not None:
            adopted = vote.adopted_mm
        else:
            adopted = self._level_mm
        level_ok = self._limits.level_ok(adopted)
        high = self._limits.level_high(adopted)
        if ok is None:
            ok = level_ok
        return LevelCheck(
            level_mm=adopted,
            target_mm=self._limits.level_target_mm,
            band_mm=self._limits.level_band_mm,
            ok=bool(ok),
            high=high,
            epoch=self._epoch,
            checked_at=self._clock.now(),
            mode=vote.mode,
            mode_name=vote.mode_name,
            divergence_mm=vote.divergence_mm,
            trusted_channel=vote.trusted_channel,
            suspect_channel=vote.suspect_channel,
            manual_release=release is not None,
            released_by=release.issued_by if release else None,
        )

    def _record_check(self, result: LevelCheck) -> None:
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
                mode=result.mode,
                mode_name=result.mode_name,
                trusted_channel=result.trusted_channel,
                suspect_channel=result.suspect_channel,
                divergence_mm=(
                    round(result.divergence_mm, 3)
                    if result.divergence_mm is not None
                    else None
                ),
                manual_release=result.manual_release,
                released_by=result.released_by,
            )
        )

    def _consume_release_if_allowed(self, vote) -> ManualRelease | None:
        """分歧态下取一张仍在有效期内的放行凭据，取出即作废（一次性）。"""

        if vote.mode != MODE_DISAGREED:
            return None
        release = self._release
        if release is None:
            return None
        if self._clock.now() > release.expires_at:
            self._release = None
            self._bus.publish(
                Event.of(
                    "furnace",
                    "furnace",
                    "level_release_expired",
                    subject="glass_level",
                    severity="warning",
                    release_id=release.release_id,
                )
            )
            return None
        self._release = None
        return release

    def _on_vote_mode_change(self, vote, previous_mode: str) -> None:
        """离开双路分歧态时，未消费的现场放行凭据自然作废。"""

        if previous_mode == MODE_DISAGREED and vote.mode != MODE_DISAGREED:
            release = self._release
            if release is not None:
                self._release = None
                self._bus.publish(
                    Event.of(
                        "furnace",
                        "furnace",
                        "level_release_voided",
                        subject="glass_level",
                        severity="notice",
                        release_id=release.release_id,
                        mode=vote.mode,
                        reason=vote.reason,
                    )
                )

    def _adopt(self, vote) -> float:
        if vote.adopted_mm is not None:
            self._level_mm = float(vote.adopted_mm)
        return self._level_mm
