from __future__ import annotations

from cli import commands


class _FakeRunner:
    def __init__(self) -> None:
        self.stop_flag_seen_at_start: bool | None = None

    def run_forever(self, stop_flag=None) -> None:
        assert stop_flag is not None
        self.stop_flag_seen_at_start = stop_flag.exists()


def test_cmd_start_clears_stale_stop_flag(monkeypatch, tmp_path):
    pid_file = tmp_path / "runner.pid"
    stop_file = tmp_path / "stop.flag"
    stop_file.write_text("stop", encoding="utf-8")

    fake_runner = _FakeRunner()
    monkeypatch.setattr(commands, "PID_FILE", pid_file)
    monkeypatch.setattr(commands, "STOP_FILE", stop_file)
    monkeypatch.setattr(commands, "setup_logging", lambda: None)
    monkeypatch.setattr(commands, "create_runner", lambda: fake_runner)

    commands.cmd_start()

    assert fake_runner.stop_flag_seen_at_start is False
    assert not stop_file.exists()
    assert not pid_file.exists()


def test_cmd_status_removes_stale_pid_file(monkeypatch, tmp_path, capsys):
    pid_file = tmp_path / "runner.pid"
    pid_file.write_text("12345", encoding="utf-8")

    monkeypatch.setattr(commands, "PID_FILE", pid_file)
    monkeypatch.setattr(commands, "_pid_running", lambda _pid: False)

    commands.cmd_status()
    output = capsys.readouterr().out

    assert "stale pid file removed" in output
    assert not pid_file.exists()


def test_cmd_status_handles_invalid_pid_file(monkeypatch, tmp_path, capsys):
    pid_file = tmp_path / "runner.pid"
    pid_file.write_text("invalid-pid", encoding="utf-8")

    monkeypatch.setattr(commands, "PID_FILE", pid_file)

    commands.cmd_status()
    output = capsys.readouterr().out

    assert "invalid pid file" in output
    assert not pid_file.exists()
