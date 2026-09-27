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

    def as_dict(self) -> dict:
        return {
            "path": self.path,
            "batch_kg": round(self.batch_kg, 3),
            "charged_kg": round(self.charged_kg, 3),
            "level_mm": round(self.level_mm, 3),
            "level_epoch": self.level_epoch,
            "level_ok": self.level_ok,
            "at": round(self.at, 3),
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
        check = level.check()
        if not check.ok:
            if check.status == "no_signal":
                reason = "两路液面测量均不可信，禁止投料，请检修测量通道"
            elif check.hold:
                reason = (
                    f"双路液面测量互相矛盾（相差 {check.diff_mm:.1f} mm），"
                    "已挂起自动投料，请到现场核实后放行"
                )
            else:
                reason = (
                    f"液面 {check.level_mm:.1f} mm 未在允许带内"
                    f"（目标 {check.target_mm:.1f} ± {check.band_mm:.1f} mm），禁止投料"
                )
            self._bus.publish(
                Event.of(
                    "batch",
                    "batch",
                    "charge_rejected",
                    subject="charger",
                    severity="warning",
                    path=path,
                    level_mm=round(check.level_mm, 3),
                    level_epoch=check.epoch,
                    high=check.high,
                    status=check.status,
                    hold=check.hold,
                    untrusted=check.untrusted,
                )
            )
            raise InterlockError(reason, component="batch")
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
                path=path,
                level_status=check.status,
                level_untrusted=check.untrusted,
                level_released=check.released,
            )
        )
        return outcome

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
