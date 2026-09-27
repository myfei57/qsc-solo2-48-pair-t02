"""控制台：页面、只读接口与写接口都走真实 HTTP。"""

from __future__ import annotations

import json
import urllib.error
import urllib.request

import pytest

from glassmelt.console.api import ConsoleAPI
from glassmelt.console.server import ConsoleServer, probe_pages


def _get(url: str) -> tuple[int, object]:
    try:
        with urllib.request.urlopen(url, timeout=5.0) as response:
            body = response.read().decode("utf-8")
            return response.status, json.loads(body) if body.startswith("{") else body
    except urllib.error.HTTPError as error:
        return error.code, json.loads(error.read().decode("utf-8"))


def _post(url: str, payload: dict | None = None) -> tuple[int, dict]:
    data = json.dumps(payload or {}).encode("utf-8")
    request = urllib.request.Request(
        url, data=data, headers={"Content-Type": "application/json"}, method="POST"
    )
    try:
        with urllib.request.urlopen(request, timeout=5.0) as response:
            return response.status, json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as error:
        return error.code, json.loads(error.read().decode("utf-8"))


@pytest.fixture()
def console(platform):
    server = ConsoleServer(platform)
    base = server.start()
    yield server, base
    server.stop()


def test_pages_and_state_endpoint_respond(console, platform):
    _server, base = console
    probe = probe_pages(base, platform.config.pages)
    assert all(item["ok"] for item in probe.values()), probe
    status, payload = _get(f"{base}/api/state")
    assert status == 200
    assert payload["platform"] == platform.config.name
    assert payload["endpoints"]["api"]


def test_operations_and_records_pages_render(console, platform):
    _server, base = console
    for page in ("operations", "records"):
        status, body = _get(f"{base}/{page}")
        assert status == 200
        assert platform.config.project_id in body
    status, body = _get(f"{base}/")
    assert status == 200
    assert "overview" in body or "总览" in body


def test_start_and_cycle_over_http(console):
    _server, base = console
    status, started = _post(f"{base}/api/start")
    assert status == 200 and started["ok"] is True
    assert started["result"]["violations"] == []
    status, cycled = _post(f"{base}/api/cycle", {"load": 0.8, "dt": 1.0})
    assert status == 200 and cycled["ok"] is True
    assert cycled["result"]["forming"]["filled"] is True


def test_action_endpoints_report_guarded_failures(console):
    _server, base = console
    status, payload = _post(f"{base}/api/batch/start", {"path": "resume"})
    assert status == 200
    assert payload["ok"] is False
    assert payload["error"]["component"] == "batch"
    status, payload = _post(f"{base}/api/mold/blow", {"path": "cycle"})
    assert status == 200 and payload["ok"] is True
    status, payload = _post(f"{base}/api/lehr/reset", {})
    assert payload["ok"] is True


def test_trend_and_readiness_interfaces(console):
    _server, base = console
    _post(f"{base}/api/start")
    _post(f"{base}/api/cycle", {"load": 0.7})
    status, trend = _get(f"{base}/api/trend?window_s=600")
    assert status == 200
    assert "melt_temp" in trend["series"]
    status, payload = _get(f"{base}/api/trend?series=unknown")
    assert status == 400
    status, readiness = _get(f"{base}/api/readiness")
    assert status == 200
    assert readiness["scale_durable"] is True


def test_audit_and_journal_interfaces(console):
    _server, base = console
    _post(f"{base}/api/start")
    status, audit = _get(f"{base}/api/audit?limit=20")
    assert status == 200
    assert audit["entries"]
    status, journal = _get(f"{base}/api/journal")
    assert status == 200
    assert journal["entries"]
    status, alarms = _get(f"{base}/api/alarms")
    assert status == 200
    assert "count" in alarms


def test_unknown_routes_are_reported(console):
    _server, base = console
    status, payload = _get(f"{base}/api/nope")
    assert status == 404
    assert payload["error"] == "not-found"
    status, payload = _post(f"{base}/api/nope")
    assert status == 404


def test_console_api_wraps_zone_and_calibration_actions(platform):
    api = ConsoleAPI(platform)
    rebuilt = api.rebuild_zone_map({"reason": "接口测试"})
    assert rebuilt["ok"] is True
    tripped = api.trip_burner({"zone": "zone-1"})
    assert tripped["ok"] is True
    assert tripped["result"]["revision"] == 2
    calibrated = api.calibrate({"offset_kg": 2.5})
    assert calibrated["result"]["fresh"] is True
    cleared = api.clear_furnace({})
    assert cleared["result"]["charger"]["interrupted"] is True


def test_level_vote_and_disagreement_release_over_http(console):
    _server, base = console
    _post(f"{base}/api/start")
    target = 1120.0
    status, payload = _post(f"{base}/api/furnace/level", {"level_mm": target + 8.0})
    assert status == 200 and payload["result"]["vote"]["mode"] == "agree"
    status, payload = _post(
        f"{base}/api/furnace/level", {"channel": "probe", "level_mm": target + 50.0}
    )
    assert payload["result"]["vote"]["mode"] == "disagreed"
    status, blocked = _post(f"{base}/api/batch/start", {"path": "start"})
    assert blocked["ok"] is False
    # 缺现场实测液位直接 400。
    status, bad = _post(f"{base}/api/furnace/level/release", {})
    assert status == 400
    # 有实测液位但未登记操作人：进入门控后被拒。
    status, rejected = _post(
        f"{base}/api/furnace/level/release", {"observed_mm": target + 6.0}
    )
    assert rejected["ok"] is False
    status, released = _post(
        f"{base}/api/furnace/level/release",
        {"issued_by": "赵工", "reason": "目视在带内，按雷达放行一批",
         "observed_mm": target + 6.0},
    )
    assert released["ok"] is True
    assert released["result"]["release"]["issued_by"] == "赵工"
    status, charged = _post(f"{base}/api/batch/start", {"path": "start"})
    assert charged["ok"] is True
    assert charged["result"]["manual_release"] is True
    status, blocked_again = _post(f"{base}/api/batch/start", {"path": "start"})
    assert blocked_again["ok"] is False


def test_level_channel_coating_and_reset_over_http(console):
    _server, base = console
    _post(f"{base}/api/start")
    target = 1120.0
    _post(f"{base}/api/furnace/level", {"level_mm": target - 10.0})
    status, payload = _post(
        f"{base}/api/furnace/level/coating",
        {"channel": "probe", "reason": "挂料挂牌"},
    )
    assert payload["result"]["mode"] == "degraded"
    assert payload["result"]["suspect_channel"] == "probe"
    _post(f"{base}/api/furnace/level/reset", {"channel": "probe", "reason": "清料"})
    _post(f"{base}/api/furnace/level", {"level_mm": target - 8.0})
    status, state = _get(f"{base}/api/state")
    assert state["level"]["vote"]["mode"] == "agree"
