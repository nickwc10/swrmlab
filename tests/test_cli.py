"""The command line: both commands, and one clear line for every kind of bad input."""

import os
import subprocess
import sys
from pathlib import Path

import pytest

from swarmsim.cli import main
from swarmsim.replay import read

ROOT = Path(__file__).parent.parent
SCENARIO = ROOT / "scenarios" / "two_outposts.yaml"
PNG_MAGIC = b"\x89PNG\r\n\x1a\n"


@pytest.fixture(scope="module")
def replay(tmp_path_factory: pytest.TempPathFactory) -> Path:
    path = tmp_path_factory.mktemp("demo") / "demo.swr"
    assert main(["demo", "--seed", "1", "--out", str(path), "--scenario", str(SCENARIO)]) == 0
    return path


def test_demo_then_plot_makes_files_in_new_folders(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.chdir(ROOT)  # the default scenario path is relative
    swr, png = tmp_path / "a" / "b" / "d.swr", tmp_path / "c" / "d.png"
    assert main(["demo", "--seed", "3", "--out", str(swr)]) == 0
    assert read(swr).header["seed"] == 3
    assert main(["plot", str(swr), "--out", str(png)]) == 0
    assert png.read_bytes().startswith(PNG_MAGIC) and png.stat().st_size > 10_000
    out = capsys.readouterr()
    assert str(swr) in out.out and str(png) in out.out and out.err == ""


def test_demo_twice_gives_identical_files(replay: Path, tmp_path: Path) -> None:
    again = tmp_path / "again.swr"
    assert main(["demo", "--seed", "1", "--out", str(again), "--scenario", str(SCENARIO)]) == 0
    assert again.read_bytes() == replay.read_bytes()


def test_the_installed_command_runs_in_two_processes_with_identical_output(tmp_path: Path) -> None:
    exe = Path(sys.executable).parent / "swarmsim"
    blobs = []
    for i, hash_seed in enumerate(("0", "1")):
        out = tmp_path / f"{i}.swr"
        done = subprocess.run(
            [str(exe), "demo", "--seed", "5", "--out", str(out)],
            cwd=ROOT,
            capture_output=True,
            text=True,
            env=os.environ | {"PYTHONHASHSEED": hash_seed},
        )
        assert (done.returncode, done.stderr) == (0, "")
        blobs.append(out.read_bytes())
    assert blobs[0] == blobs[1]


def _fails(args: list[str], capsys: pytest.CaptureFixture[str], expect: str) -> None:
    try:
        code = main(args)
    except SystemExit as stop:  # argparse exits by itself
        code = int(stop.code or 0)
    captured = capsys.readouterr()
    assert code != 0
    assert captured.out == ""
    lines = captured.err.splitlines()
    assert len(lines) == 1, captured.err
    assert lines[0].startswith("swarmsim: error: ")
    assert expect in lines[0] and "Traceback" not in lines[0]


@pytest.mark.parametrize("seed", ["-1", str(2**63), "abc", "1.5", ""])
def test_bad_seed(seed: str, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    out = tmp_path / "x.swr"
    args = ["demo", f"--seed={seed}", "--out", str(out), "--scenario", str(SCENARIO)]
    _fails(args, capsys, "seed")
    assert not out.exists()


@pytest.mark.parametrize(
    ("text", "expect"),
    [
        (None, "not found"),
        ("a: [1, 2", "invalid YAML"),
        ("- just a list", "mapping"),
        ("name: x\nship_classes: {}\nblue: {}\nred: {}\nmap: {home_rocks: -5}\n", "home_rocks"),
        ("name: x\nbogus: 1\nship_classes: {}\nblue: {}\nred: {}\n", "bogus"),
        (b"\xff\xfe\x00", "bad.yaml"),
        (
            "name: x\nship_classes: {}\nblue: {}\nred: {}\n"
            "map: {cluster_radius: 100000.0, home_rocks: 5000}\n",
            "could not place",
        ),
    ],
)
def test_bad_scenario(
    text: str | bytes | None, expect: str, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    path = tmp_path / "bad.yaml"
    if text is not None:
        path.write_bytes(text.encode() if isinstance(text, str) else text)
    out = tmp_path / "new" / "x.swr"
    _fails(["demo", "--seed", "1", "--out", str(out), "--scenario", str(path)], capsys, expect)
    assert not out.parent.exists()


def test_scenario_is_a_folder(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    args = ["demo", "--seed", "1", "--out", str(tmp_path / "x"), "--scenario", str(tmp_path)]
    _fails(args, capsys, str(tmp_path))


def test_missing_replay(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    missing = tmp_path / "none.swr"
    _fails(["plot", str(missing), "--out", str(tmp_path / "x.png")], capsys, str(missing))


def test_damaged_replay(replay: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    blob = bytearray(replay.read_bytes())
    blob[len(blob) // 2] ^= 1
    bad, out = tmp_path / "bad.swr", tmp_path / "x.png"
    for content, expect in ((bytes(blob), "checksum"), (b"", "shorter"), (b"SWR2" * 20, "bad.swr")):
        bad.write_bytes(content)
        _fails(["plot", str(bad), "--out", str(out)], capsys, expect)
        assert not out.exists()


def test_unwritable_output(
    replay: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    blocker = tmp_path / "file"
    blocker.write_text("in the way")
    for out in (blocker / "x", tmp_path):  # a folder that cannot be made; a folder as the file
        args = ["demo", "--seed", "1", "--out", str(out), "--scenario", str(SCENARIO)]
        _fails(args, capsys, str(out))
        _fails(["plot", str(replay), "--out", str(out)], capsys, str(out))


def test_read_only_folder(replay: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    locked = tmp_path / "locked"
    locked.mkdir()
    locked.chmod(0o500)
    try:
        if os.access(locked, os.W_OK):
            pytest.skip("running as a user that ignores folder permissions")
        out = str(locked / "x")
        args = ["demo", "--seed", "1", "--out", out, "--scenario", str(SCENARIO)]
        _fails(args, capsys, "locked")
        _fails(["plot", str(replay), "--out", out], capsys, "locked")
    finally:
        locked.chmod(0o700)


def test_plot_refuses_to_overwrite_its_own_replay(
    replay: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    before = replay.read_bytes()
    _fails(["plot", str(replay), "--out", str(replay)], capsys, "replay")
    assert replay.read_bytes() == before


@pytest.mark.parametrize("spelling", ["same", "dotdot", "symlink", "default"])
def test_demo_refuses_to_overwrite_its_own_scenario(
    spelling: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    folder = tmp_path / "scenarios"
    folder.mkdir()
    copy = folder / "two_outposts.yaml"
    copy.write_bytes(SCENARIO.read_bytes())
    monkeypatch.chdir(tmp_path)
    link = tmp_path / "link.swr"
    link.symlink_to(copy)
    out = {
        "same": str(copy),
        "dotdot": str(folder / ".." / "scenarios" / "two_outposts.yaml"),
        "symlink": str(link),
        "default": "scenarios/two_outposts.yaml",
    }[spelling]
    args = ["demo", "--seed", "1", "--out", out]
    if spelling != "default":
        args += ["--scenario", str(copy)]
    _fails(args, capsys, "scenario")
    assert copy.read_bytes() == SCENARIO.read_bytes()
    assert link.is_symlink()


def test_demo_writes_a_new_file_beside_the_scenario(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    copy, out = tmp_path / "s.yaml", tmp_path / "s.swr"
    copy.write_bytes(SCENARIO.read_bytes())
    args = ["demo", "--seed", "1", "--out", str(out), "--scenario", str(copy)]
    assert main(args) == 0
    first = out.read_bytes()
    assert main(args) == 0  # replacing an earlier replay is fine
    assert out.read_bytes() == first
    assert copy.read_bytes() == SCENARIO.read_bytes()
    assert capsys.readouterr().err == ""


def test_no_command_or_missing_option(capsys: pytest.CaptureFixture[str]) -> None:
    _fails([], capsys, "command")
    _fails(["demo", "--seed", "1"], capsys, "--out")
