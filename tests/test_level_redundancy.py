"""双路液面校验：一致采信、单路坏降级、两路打架挂起等现场放行。"""

from __future__ import annotations

import pytest

from glassmelt.app import CycleInputs
from glassmelt.errors import InterlockError, SequenceError


def test_agreeing_channels_are_trusted(platform):
    target = platform.config.furnace.level_target_mm
    platform.level.observe(target)
    check = platform.level.check()
    assert check.ok is True
    assert check.status == "agreed"
    assert check.channel == "both"
    assert check.untrusted is None
    snapshot = platform.level.snapshot()
    assert snapshot["channels"]["probe"]["trusted"] is True
    assert snapshot["channels"]["radar"]["trusted"] is True
    assert snapshot["channels"]["probe"]["principle"] != snapshot["channels"]["radar"]["principle"]


def test_implausible_reading_degrades_to_the_other_channel(platform):
    target = platform.config.furnace.level_target_mm
    platform.level.observe(target)
    platform.inject_level_reading("radar", platform.config.furnace.level_plausible_max_mm + 1.0)
    check = platform.level.check()
    assert check.ok is True
    assert check.status == "degraded"
    assert check.untrusted == "radar"
    assert check.level_mm == pytest.approx(target)
    snapshot = platform.level.snapshot()
    assert snapshot["channels"]["radar"]["trusted"] is False
    assert snapshot["channels"]["radar"]["reason"]
    outcome = platform.start_charging(path="start")
    assert outcome["level_ok"] is True


def test_transmitter_fault_bit_marks_channel_untrusted(platform):
    target = platform.config.furnace.level_target_mm
    platform.level.observe(target)
    platform.flag_level_channel("probe", True, "结料")
    check = platform.level.check()
    assert check.status == "degraded"
    assert check.untrusted == "probe"
    assert check.ok is True
    platform.flag_level_channel("probe", False)
    assert platform.level.check().status == "agreed"


def test_frozen_reading_is_rejected(platform):
    target = platform.config.furnace.level_target_mm
    platform.level.observe(target)
    frozen = platform.config.furnace.level_frozen_count
    for _ in range(frozen):
        platform.inject_level_reading("radar", target - 10.0)
    snapshot = platform.level.snapshot()
    assert snapshot["channels"]["radar"]["trusted"] is False
    assert "冻结" in snapshot["channels"]["radar"]["reason"]
    assert platform.level.check().status == "degraded"


def test_disagreeing_channels_hold_charging(platform):
    target = platform.config.furnace.level_target_mm
    agree = platform.config.furnace.level_agree_mm
    platform.level.observe(target)
    platform.inject_level_reading("probe", target + agree + 10.0)
    check = platform.level.check()
    assert check.ok is False
    assert check.hold is True
    assert check.status == "disagree"
    assert check.diff_mm == pytest.approx(agree + 10.0)
    with pytest.raises(InterlockError, match="现场核实"):
        platform.start_charging(path="start")
    assert platform.level.snapshot()["hold"]["active"] is True


def test_manual_release_lets_charging_resume(platform):
    target = platform.config.furnace.level_target_mm
    agree = platform.config.furnace.level_agree_mm
    platform.level.observe(target)
    platform.inject_level_reading("probe", target + agree + 10.0)
    with pytest.raises(InterlockError):
        platform.start_charging(path="start")
    released = platform.release_level_hold(note="现场看液面正常，探针结料", trust="radar")
    assert released["trust"] == "radar"
    outcome = platform.start_charging(path="start")
    assert outcome["level_ok"] is True
    confirmation = platform.level.last_confirmation()
    assert confirmation.released is True
    assert confirmation.note == "现场看液面正常，探针结料"


def test_release_is_scoped_to_the_disagreement_episode(platform):
    target = platform.config.furnace.level_target_mm
    agree = platform.config.furnace.level_agree_mm
    platform.level.observe(target)
    platform.inject_level_reading("probe", target + agree + 10.0)
    platform.release_level_hold(note="第一轮放行", trust="radar")
    assert platform.level.check().ok is True
    platform.inject_level_reading("probe", target)  # 探头修复，分歧消除
    assert platform.level.snapshot()["hold"]["active"] is False
    platform.inject_level_reading("probe", target + agree + 10.0)
    check = platform.level.check()
    assert check.hold is True
    assert check.ok is False
    with pytest.raises(InterlockError):
        platform.start_charging(path="start")


def test_release_without_hold_is_rejected(platform):
    platform.level.observe(platform.config.furnace.level_target_mm)
    with pytest.raises(InterlockError):
        platform.release_level_hold(note="没有持留却放行")


def test_release_with_unknown_channel_is_rejected(platform):
    target = platform.config.furnace.level_target_mm
    platform.level.observe(target)
    platform.inject_level_reading("probe", target + platform.config.furnace.level_agree_mm + 5.0)
    with pytest.raises(SequenceError):
        platform.release_level_hold(note="放行到不存在的通道", trust="sonar")


def test_both_channels_bad_blocks_charging(platform):
    target = platform.config.furnace.level_target_mm
    platform.level.observe(target)
    platform.flag_level_channel("probe", True, "结料")
    platform.flag_level_channel("radar", True, "信号丢失")
    check = platform.level.check()
    assert check.ok is False
    assert check.status == "no_signal"
    assert check.untrusted == "probe+radar"
    with pytest.raises(InterlockError, match="均不可信"):
        platform.start_charging(path="start")


def test_clear_furnace_voids_the_release(platform):
    target = platform.config.furnace.level_target_mm
    agree = platform.config.furnace.level_agree_mm
    platform.level.observe(target)
    platform.inject_level_reading("probe", target + agree + 10.0)
    platform.release_level_hold(note="放行", trust="radar")
    platform.clear_furnace("出料")
    check = platform.level.check()
    assert check.hold is True
    assert check.ok is False


def test_channel_rejection_and_release_are_audited(platform):
    target = platform.config.furnace.level_target_mm
    agree = platform.config.furnace.level_agree_mm
    platform.level.observe(target)
    platform.inject_level_reading("probe", platform.config.furnace.level_plausible_max_mm + 1.0)
    platform.inject_level_reading("probe", target + agree + 10.0)
    platform.release_level_hold(note="现场核实", trust="radar")
    actions = [
        entry["action"] for entry in platform.audit.entries(200) if entry["component"] == "furnace"
    ]
    assert "level_channel_rejected" in actions
    assert "level_channel_recovered" in actions
    assert "level_hold_raised" in actions
    assert "level_hold_released" in actions


def test_cycle_holds_charging_while_channels_disagree(running):
    target = running.config.furnace.level_target_mm
    agree = running.config.furnace.level_agree_mm
    running.level.observe(target)
    running.inject_level_reading("probe", target + agree + 10.0)
    before = running.charger.charged_kg
    summary = running.cycle(CycleInputs(load=0.8, pull_rate=0.9, gob_mass_g=186.0, dt=1.0))
    assert summary["charge"]["charged_kg"] == 0.0
    assert running.charger.charged_kg == before
    assert summary["level"]["hold"]["active"] is True
    running.release_level_hold(note="现场核实，信雷达", trust="radar")
    radar_mm = running.level.snapshot()["channels"]["radar"]["value_mm"]
    running.inject_level_reading("probe", radar_mm)  # 探头修复，分歧消除
    assert running.level.snapshot()["hold"]["active"] is False
    summary = running.cycle(CycleInputs(load=0.8, pull_rate=0.9, gob_mass_g=186.0, dt=1.0))
    assert summary["level"]["status"] == "agreed"
