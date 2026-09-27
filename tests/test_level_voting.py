"""双路液面互校门控。

覆盖用户要的四条裁决：两路一致才采信；单路坏用剩下那路并标明不可信的
一路；两路打架停自动投料、现场确认后一次性放行；两路全盲禁止投料。同时
覆盖共同跳变不误判、冻结结料识别、放行凭据的一次性与有效期。
"""

from __future__ import annotations

import pytest

from glassmelt.errors import InterlockError, SequenceError
from glassmelt.furnace.levelvote import (
    CHANNEL_PROBE,
    CHANNEL_RADAR,
    MODE_AGREE,
    MODE_BLIND,
    MODE_DEGRADED,
    MODE_DISAGREED,
)


def _target(platform) -> float:
    return platform.config.furnace.level_target_mm


def test_initial_state_is_blind_until_both_channels_report(platform):
    vote = platform.level.snapshot()["vote"]
    assert vote["mode"] == MODE_BLIND
    assert vote["trusted"] is False


def test_two_channels_in_agreement_are_trusted(platform):
    platform.level.observe(_target(platform) - 10.0)
    vote = platform.level.snapshot()["vote"]
    assert vote["mode"] == MODE_AGREE
    assert vote["trusted"] is True
    assert vote["adopted_mm"] == pytest.approx(_target(platform) - 10.0)
    assert platform.level.channel(CHANNEL_PROBE).raw_mm == pytest.approx(
        _target(platform) - 10.0
    )


def test_small_natural_difference_still_agrees(platform):
    platform.level.observe_channel(CHANNEL_PROBE, _target(platform))
    platform.level.observe_channel(CHANNEL_RADAR, _target(platform) + 5.0)
    assert platform.level.snapshot()["vote"]["mode"] == MODE_AGREE


def test_simultaneous_equal_jump_is_a_real_level_change_not_a_spike(platform):
    # 前天白停料的场景：两路同一拍跳同样的幅度（出料/塌料），不能停料。
    platform.level.observe(_target(platform))
    platform.level.observe(_target(platform) + 90.0)
    vote = platform.level.snapshot()["vote"]
    assert vote["mode"] == MODE_AGREE
    assert platform.level.channel(CHANNEL_PROBE).fault is None
    assert platform.level.channel(CHANNEL_RADAR).fault is None


def test_single_channel_spike_is_latched_and_other_channel_runs(platform):
    platform.level.observe(_target(platform) - 20.0)
    # 只有探针跳变，雷达不动：探针被判坏，门控降级采信雷达。
    platform.level.observe_channel(CHANNEL_PROBE, _target(platform) + 80.0)
    vote = platform.level.snapshot()["vote"]
    assert vote["mode"] == MODE_DEGRADED
    assert vote["trusted"] is True
    assert vote["trusted_channel"] == CHANNEL_RADAR
    assert vote["suspect_channel"] == CHANNEL_PROBE
    assert platform.level.channel(CHANNEL_PROBE).fault == "spike"
    # 降级态下雷达显示液面在带内，投料仍可进行，且记录标明哪一路不可信。
    outcome = platform.start_charging(path="start")
    assert outcome["level_mode"] == MODE_DEGRADED
    assert outcome["trusted_channel"] == CHANNEL_RADAR
    assert outcome["suspect_channel"] == CHANNEL_PROBE
    assert outcome["level_mm"] == pytest.approx(_target(platform) - 20.0)


def test_coating_mark_takes_probe_out_and_radar_keeps_feeding(platform):
    platform.level.observe(_target(platform) - 15.0)
    platform.mark_level_channel_coating(CHANNEL_PROBE, "探针结料挂料")
    vote = platform.level.snapshot()["vote"]
    assert vote["mode"] == MODE_DEGRADED
    assert vote["suspect_channel"] == CHANNEL_PROBE
    outcome = platform.start_charging(path="start")
    assert outcome["suspect_channel"] == CHANNEL_PROBE
    assert outcome["trusted_channel"] == CHANNEL_RADAR


def test_disagreement_blocks_auto_charge(platform):
    platform.level.observe(_target(platform) + 8.0)
    platform.level.observe_channel(CHANNEL_PROBE, _target(platform) + 50.0)
    vote = platform.level.snapshot()["vote"]
    assert vote["mode"] == MODE_DISAGREED
    assert vote["trusted"] is False
    with pytest.raises(InterlockError):
        platform.start_charging(path="start")
    # 被门控挡住时不得产生投料批次。
    assert platform.charger.batch_count == 0


def test_release_requires_operator_and_reason(platform):
    platform.level.observe(_target(platform) + 8.0)
    platform.level.observe_channel(CHANNEL_PROBE, _target(platform) + 50.0)
    with pytest.raises(InterlockError):
        platform.release_level_disagreement("", "有人看过", _target(platform))
    with pytest.raises(InterlockError):
        platform.release_level_disagreement("赵工", "", _target(platform))


def test_release_not_allowed_unless_actually_disagreed(platform):
    platform.level.observe(_target(platform) - 5.0)
    with pytest.raises(InterlockError):
        platform.release_level_disagreement("赵工", "两路好好的，想强行放行", _target(platform))


def test_site_release_allows_exactly_one_charge(platform):
    platform.level.observe(_target(platform) + 8.0)
    platform.level.observe_channel(CHANNEL_PROBE, _target(platform) + 50.0)
    platform.release_level_disagreement("赵工", "现场目视在带内，探针虚高", _target(platform) + 6.0)
    outcome = platform.start_charging(path="start")
    assert outcome["manual_release"] is True
    assert outcome["released_by"] == "赵工"
    assert outcome["level_mode"] == MODE_DISAGREED
    # 放行凭据一次性：第二拍仍分歧时不能再自动投料。
    with pytest.raises(InterlockError):
        platform.start_charging(path="start")
    assert platform.charger.batch_count == 1


def test_release_expires_after_the_window(platform, clock):
    platform.level.observe(_target(platform) + 8.0)
    platform.level.observe_channel(CHANNEL_PROBE, _target(platform) + 50.0)
    platform.release_level_disagreement("赵工", "现场目视在带内", _target(platform) + 6.0)
    clock.advance(platform.config.furnace.level_release_window_s + 1.0)
    with pytest.raises(InterlockError):
        platform.start_charging(path="start")
    assert platform.charger.batch_count == 0


def test_release_is_voided_when_channels_agree_again(platform):
    platform.level.observe(_target(platform) + 8.0)
    platform.level.observe_channel(CHANNEL_PROBE, _target(platform) + 50.0)
    platform.release_level_disagreement("赵工", "现场目视在带内", _target(platform) + 6.0)
    assert platform.level.pending_release() is not None
    # 两路在放行前自行恢复一致：未消费的旧凭据自然作废，后续按双路一致走。
    platform.level.observe(_target(platform) - 6.0)
    assert platform.level.pending_release() is None
    outcome = platform.start_charging(path="start")
    assert outcome["level_mode"] == MODE_AGREE
    assert outcome["manual_release"] is False


def test_blind_state_blocks_charge_and_even_site_release(platform):
    platform.level.observe(_target(platform) - 10.0)
    platform.mark_level_channel_coating(CHANNEL_PROBE)
    platform.mark_level_channel_coating(CHANNEL_RADAR)
    vote = platform.level.snapshot()["vote"]
    assert vote["mode"] == MODE_BLIND
    with pytest.raises(InterlockError):
        platform.start_charging(path="start")
    with pytest.raises(InterlockError):
        platform.release_level_disagreement("赵工", "全盲也想放行", _target(platform))


def test_channel_reset_rejoins_voting(platform):
    platform.level.observe(_target(platform) - 12.0)
    platform.level.observe_channel(CHANNEL_PROBE, _target(platform) + 80.0)
    assert platform.level.snapshot()["vote"]["mode"] == MODE_DEGRADED
    result = platform.reset_level_channel(CHANNEL_PROBE, "清挂料后复位")
    # 复位后探针清空旧样本，单雷达仍是降级；重新送样后恢复双路一致。
    assert result["cleared_fault"] == "spike"
    platform.level.observe(_target(platform) - 8.0)
    assert platform.level.snapshot()["vote"]["mode"] == MODE_AGREE


def test_frozen_probe_with_drifting_radar_is_detected_as_stuck(platform, clock):
    target = _target(platform)
    platform.level.observe(target - 8.0)
    # 探针结料挂住：之后每拍只有雷达随出料缓慢下降，探针一直不动。
    span = platform.config.furnace.level_stuck_span_s
    steps = 6
    for index in range(1, steps + 1):
        clock.advance(span / steps)
        radar_value = target - 8.0 - 3.0 * index
        platform.level.observe_channel(CHANNEL_RADAR, radar_value)
    vote = platform.level.snapshot()["vote"]
    assert vote["mode"] == MODE_DEGRADED
    assert vote["suspect_channel"] == CHANNEL_PROBE
    assert platform.level.channel(CHANNEL_PROBE).fault == "stuck"
    assert vote["trusted_channel"] == CHANNEL_RADAR


def test_range_fault_latches_a_channel(platform):
    platform.level.observe(_target(platform) - 10.0)
    platform.level.observe_channel(CHANNEL_RADAR, platform.config.furnace.level_max_mm + 50.0)
    vote = platform.level.snapshot()["vote"]
    assert vote["mode"] == MODE_DEGRADED
    assert vote["suspect_channel"] == CHANNEL_RADAR
    assert platform.level.channel(CHANNEL_RADAR).fault == "range"


def test_stale_channel_degrades_then_self_heals(platform, clock):
    target = _target(platform)
    platform.level.observe(target - 6.0)
    # 雷达超时不刷新：瞬态失联（不锁故障），降级采信探针。
    clock.advance(platform.config.furnace.level_stale_s + 1.0)
    # 触发一次重判（探针本拍仍正常送样）。
    platform.level.observe_channel(CHANNEL_PROBE, target - 6.0)
    vote = platform.level.snapshot()["vote"]
    assert vote["mode"] == MODE_DEGRADED
    assert vote["trusted_channel"] == CHANNEL_PROBE
    assert platform.level.channel(CHANNEL_RADAR).fault is None
    # 雷达恢复送样，无需维护复位即回到双路一致。
    platform.level.observe(target - 5.0)
    assert platform.level.snapshot()["vote"]["mode"] == MODE_AGREE


def test_clear_furnace_voids_pending_release(platform):
    platform.level.observe(_target(platform) + 8.0)
    platform.level.observe_channel(CHANNEL_PROBE, _target(platform) + 50.0)
    platform.release_level_disagreement("赵工", "现场目视在带内", _target(platform) + 6.0)
    platform.clear_furnace("出料")
    assert platform.level.pending_release() is None
    assert platform.level.last_confirmation() is None


def test_unknown_channel_is_rejected(platform):
    with pytest.raises(SequenceError):
        platform.observe_level(1100.0, channel="ultrasound")
    with pytest.raises(SequenceError):
        platform.reset_level_channel("ultrasound")


def test_snapshot_exposes_both_channels_and_vote(platform):
    platform.level.observe(_target(platform) - 6.0)
    snapshot = platform.level.snapshot()
    assert set(snapshot["channels"]) == {CHANNEL_PROBE, CHANNEL_RADAR}
    assert snapshot["channels"][CHANNEL_PROBE]["healthy"] is True
    assert snapshot["vote"]["mode"] == MODE_AGREE
    assert snapshot["releases"] >= 0


def test_fault_can_be_raised_again_after_maintenance_reset(platform):
    target = _target(platform)
    platform.level.observe(target - 6.0)
    platform.level.observe_channel(CHANNEL_PROBE, target + 60.0)
    assert platform.level.channel(CHANNEL_PROBE).fault == "spike"
    platform.reset_level_channel(CHANNEL_PROBE)
    platform.level.observe(target - 6.0)
    # 复位后探针再次单点跳变，必须重新锁定并再次上报告警。
    platform.level.observe_channel(CHANNEL_PROBE, target + 60.0)
    assert platform.level.channel(CHANNEL_PROBE).fault == "spike"
    faults = [
        event
        for event in platform.bus.history("furnace")
        if event.action == "level_channel_fault" and event.payload["channel"] == CHANNEL_PROBE
    ]
    assert len(faults) == 2


def test_release_with_out_of_band_observed_level_is_rejected(platform):
    # 现场实测液位本身已在允许带以上：即便人到现场，也不允许签发放行投料。
    target = _target(platform)
    platform.level.observe_channel(CHANNEL_PROBE, target + 60.0)
    platform.level.observe_channel(CHANNEL_RADAR, target + 80.0)
    assert platform.level.snapshot()["vote"]["mode"] == MODE_DISAGREED
    with pytest.raises(InterlockError):
        platform.release_level_disagreement("赵工", "现场实测偏高", target + 70.0)
    assert platform.charger.batch_count == 0


def test_release_uses_observed_level_for_the_gate_and_record(platform):
    target = _target(platform)
    # 探针虚高、雷达在带内：人到现场按实测液位放行，记录里保留该实测值。
    platform.level.observe(target + 8.0)
    platform.level.observe_channel(CHANNEL_PROBE, target + 50.0)
    observed = target + 6.0
    platform.release_level_disagreement("赵工", "探针虚高，按实测放行", observed)
    outcome = platform.start_charging(path="start")
    assert outcome["level_mm"] == pytest.approx(observed)
    assert outcome["level_ok"] is True
    assert platform.level.last_confirmation().released_by == "赵工"
