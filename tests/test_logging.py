import importlib.util
import os
import sys

from agentic_tactile_vla.recorder import Recorder


def _load_run_agent():
    path = os.path.join(os.path.dirname(__file__), "..", "scripts", "run_agent.py")
    spec = importlib.util.spec_from_file_location("run_agent", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_session_log_and_csv_share_the_stamp(tmp_path, monkeypatch):
    monkeypatch.setenv("AGENT_LOG_DIR", str(tmp_path))
    monkeypatch.delenv("AGENT_SESSION_STAMP", raising=False)
    monkeypatch.setattr(sys, "stdout", sys.stdout)
    monkeypatch.setattr(sys, "stderr", sys.stderr)
    stamp = _load_run_agent()._start_session_log("left")
    print("hello from the bridge")
    sys.stdout.flush()
    rec = Recorder(hand="left")
    rec.close()
    names = sorted(os.listdir(tmp_path))
    assert names == [f"bridge_left_{stamp}.log", f"ramp_left_{stamp}.csv",
                     f"verdicts_left_{stamp}.csv"]
    assert "hello from the bridge" in (tmp_path / f"bridge_left_{stamp}.log").read_text()


def test_a_second_session_does_not_overwrite(tmp_path, monkeypatch):
    monkeypatch.setenv("AGENT_LOG_DIR", str(tmp_path))
    (tmp_path / "bridge_left_20260101_000000.log").write_text("earlier session")
    monkeypatch.setattr(sys, "stdout", sys.stdout)
    monkeypatch.setattr(sys, "stderr", sys.stderr)
    _load_run_agent()._start_session_log("left")
    assert (tmp_path / "bridge_left_20260101_000000.log").read_text() == "earlier session"
