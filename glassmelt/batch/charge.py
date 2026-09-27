"""投料机。

投料机有两条进入路径：正常启动与联锁恢复。两条路径都必须先拿到一次现场
液面复核，确认液面没到允许带以上才允许下料——只堵住其中一条，物料仍会
在另一条路径上堆过熔化线。
"""

from __future__ import annotations

import threading
from dataclasses import dataclass

from ..config import BatchLimits
from ..errors import InterlockError
from ..eventbus import Event, EventBus


@dataclass(frozen=True)
class ChargeOutcome:
    """一次投料动作的结果。"""

    path: str
    batch_kg: float
    charged_kg: float
    level_mm: float
    level_epoch: int
    level_ok: bool
    at: float
    level_mode: str = "agree"
    level_mode_name: str = "双路一致"
    trusted_channel: str | None = None
    suspect_channel: str | None = None
    divergence_mm: float | None = None
    manual_release: bool = False
    released_by: str | None = None

    def as_dict(self) -> dict:
        return {
            "path": self.path,
            "batch_kg": round(self.batch_kg, 3),
            "charged_kg": round(self.charged_kg, 3),
            "level_mm": round(self.level_mm, 3),
            "level_epoch": self.level_epoch,
            "level_ok": self.level_ok,
            "at": round(self.at, 3),
            "level_mode": self.level_mode,
            "level_mode_name": self.level_mode_name,
            "trusted_channel": self.trusted_channel,
            "suspect_channel": self.suspect_channel,
            "divergence_mm": (
                round(self.divergence_mm, 3) if self.divergence_mm is not None else None
            ),
            "manual_release": self.manual_release,
            "released_by": self.released_by,
        }


class Charger:
    """投料机本体。"""

    def __init__(self, bus: EventBus, limits: BatchLimits) -> None:
        self._bus = bus
        self._limits = limits
        self._charged_kg = 0.0
        self._batch_count = 0
        self._interrupted = False
        self._history: list[dict] = []
        self._lock = threading.RLock()

    @property
    def charged_kg(self) -> float:
        with self._lock:
            return self._charged_kg

    @property
    def batch_count(self) -> int:
        with self._lock:
            return self._batch_count

    @property
    def interrupted(self) -> bool:
        with self._lock:
            return self._interrupted

    def start(self, level) -> ChargeOutcome:
        """正常启动投料：先做一次现场液面复核。"""

        return self._charge(level, path="start")

    def resume(self, level) -> ChargeOutcome:
        """联锁恢复后继续投料：同样要先做一次现场液面复核。"""

        with self._lock:
            if not self._interrupted:
                raise InterlockError("投料机当前没有待恢复的联锁", component="batch")
        outcome = self._charge(level, path="resume")
        with self._lock:
            self._interrupted = False
        return outcome

    def interrupt(self, reason: str) -> None:
        """投料联锁动作：置位待恢复标记。"""

        with self._lock:
            self._interrupted = True
        self._bus.publish(
            Event.of(
                "batch",
                "batch",
                "charge_interrupted",
                subject="charger",
                severity="warning",
                reason=reason,
            )
        )

    def _charge(self, level, path: str) -> ChargeOutcome:
        try:
            check = level.check()
        except InterlockError as error:
            # 双路分歧／双盲时确认本身就拒绝；同样记一笔投料拒绝再上抛。
            self._publish_rejected(path, level.last_confirmation(), reason=error.message)
            raise
        if not check.ok:
            self._publish_rejected(path, check)
            raise InterlockError(
                f"液面 {check.level_mm:.1f} mm 未在允许带内"
                f"（目标 {check.target_mm:.1f} ± {check.band_mm:.1f} mm），禁止投料",
                component="batch",
            )
        batch_kg = self._limits.charge_batch_kg
        with self._lock:
            self._charged_kg += batch_kg
            self._batch_count += 1
            outcome = ChargeOutcome(
                path=path,
                batch_kg=batch_kg,
                charged_kg=self._charged_kg,
                level_mm=check.level_mm,
                level_epoch=check.epoch,
                level_ok=check.ok,
                at=check.checked_at,
                level_mode=check.mode,
                level_mode_name=check.mode_name,
                trusted_channel=check.trusted_channel,
                suspect_channel=check.suspect_channel,
                divergence_mm=check.divergence_mm,
                manual_release=check.manual_release,
                released_by=check.released_by,
            )
            self._history.append(outcome.as_dict())
            if len(self._history) > 200:
                del self._history[: len(self._history) - 200]
        self._bus.publish(
            Event.of(
                "batch",
                "batch",
                "charged",
                subject="charger",
                severity="notice",
                batch_kg=round(batch_kg, 3),
                level_mm=round(check.level_mm, 3),
                level_epoch=check.epoch,
                level_mode=check.mode,
                trusted_channel=check.trusted_channel,
                suspect_channel=check.suspect_channel,
                manual_release=check.manual_release,
                released_by=check.released_by,
                path=path,
            )
        )
        return outcome

    def _publish_rejected(self, path: str, check, reason: str | None = None) -> None:
        level_mm = check.level_mm if check is not None else None
        epoch = check.epoch if check is not None else None
        mode = check.mode if check is not None else None
        self._bus.publish(
            Event.of(
                "batch",
                "batch",
                "charge_rejected",
                subject="charger",
                severity="warning",
                path=path,
                level_mm=round(level_mm, 3) if level_mm is not None else None,
                level_epoch=epoch,
                level_mode=mode,
                high=check.high if check is not None else None,
                reason=reason,
            )
        )

    def history(self, limit: int = 5) -> list[dict]:
        with self._lock:
            return self._history[-limit:]

    def snapshot(self) -> dict:
        with self._lock:
            return {
                "charged_kg": round(self._charged_kg, 3),
                "batch_count": self._batch_count,
                "interrupted": self._interrupted,
                "batch_kg": self._limits.charge_batch_kg,
                "recent": self.history(),
            }
