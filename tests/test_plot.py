"""The plot: built from the replay file alone, with light delay from the light rules."""

from pathlib import Path
from typing import Any

import numpy as np
import pytest

from swarmsim.core.light import emission_time
from swarmsim.core.segments import SegmentStore
from swarmsim.demo import run_demo
from swarmsim.plot import GROUND, TEAM_STYLE, build_figure, light_gap, plot, separations
from swarmsim.replay import Replay, read, write
from swarmsim.scenario import Scenario, build_world, load

SCENARIO = Path(__file__).parent.parent / "scenarios" / "two_outposts.yaml"


def _save(path: Path, world: Any, scenario: Scenario, seed: int) -> Replay:
    write(path, world, scenario, seed)
    return read(path)


@pytest.fixture(scope="module")
def demo(tmp_path_factory: pytest.TempPathFactory) -> tuple[Replay, SegmentStore]:
    scenario = load(SCENARIO)
    world = run_demo(scenario, 1)
    return _save(tmp_path_factory.mktemp("r") / "d.swr", world, scenario, 1), world.store


def test_light_gap_equals_the_light_rules_on_the_original_run(
    demo: tuple[Replay, SegmentStore],
) -> None:
    replay, store = demo
    times = [2_000_000, 10_000_000_000, 20_000_000_001, 30_123_456_789, 100_000_000_000]
    got = light_gap(replay, 0, 6, times)
    for t, gap in zip(times, got, strict=True):
        here = store.state(0, t)[0]
        seen = emission_time(store, 6, t, here, replay.scenario.c)
        assert seen is not None
        true = float(np.linalg.norm(store.state(6, t)[0] - here))
        delayed = float(np.linalg.norm(store.state(6, seen)[0] - here))
        assert gap == true - delayed
    # A closing ship is really nearer than it looks, by about speed x delay.
    assert -20e3 < got[1] < -5e3


def test_light_gap_is_missing_before_first_light(demo: tuple[Replay, SegmentStore]) -> None:
    gaps = light_gap(demo[0], 0, 6, [0, 900_000, 1_100_000])
    assert np.isnan(gaps[:2]).all() and np.isfinite(gaps[2])


def test_light_gap_uses_the_scenario_light_speed(tmp_path: Path) -> None:
    slow = load(SCENARIO).model_copy(update={"c": 3e7})
    slow = Scenario.model_validate(slow.model_dump())
    replay = _save(tmp_path / "s.swr", run_demo(slow, 1), slow, 1)
    base = Scenario.model_validate(load(SCENARIO).model_dump())
    normal = _save(tmp_path / "n.swr", run_demo(base, 1), base, 1)
    t = [15_000_000_000]
    ratio = light_gap(replay, 0, 6, t)[0] / light_gap(normal, 0, 6, t)[0]
    assert 8.0 < ratio < 12.0


def test_light_gap_stops_when_the_ship_ends(tmp_path: Path) -> None:
    scenario = load(SCENARIO)
    world = build_world(scenario, 1)
    rock = world.rocks.centers[0]
    aim = rock - world.store.state(6, 0)[0]
    world.command_acc(6, 0, aim / np.linalg.norm(aim))
    world.run_until(world.horizon_us)
    end = world.store.end_time(6)
    assert end is not None
    replay = _save(tmp_path / "e.swr", world, scenario, 1)
    gaps = light_gap(replay, 0, 6, [end - 1_000_000, end + 1, end + 10**7])
    assert np.isfinite(gaps[0]) and np.isnan(gaps[1:]).all()
    assert build_figure(replay) is not None


def test_each_outpost_watches_the_first_enemy_ship(demo: tuple[Replay, SegmentStore]) -> None:
    assert separations(demo[0]) == [("blue", 0, 6), ("red", 5, 1)]


def test_figure_has_both_panels_with_units(demo: tuple[Replay, SegmentStore]) -> None:
    fig = build_figure(demo[0])
    top, bottom = fig.axes
    assert "thousand km" in top.get_xlabel() and "thousand km" in top.get_ylabel()
    assert "stretched" in top.get_title(loc="left") + "".join(t.get_text() for t in top.texts)
    assert "km" in bottom.get_ylabel() and "hours" in bottom.get_xlabel()
    assert len([ln for ln in bottom.lines if ln.get_gid() == "gap"]) == 2
    assert len([ln for ln in top.lines if ln.get_gid() == "path"]) == 8
    for ax in (top, bottom):
        assert ax.get_legend() is not None
    labels = " ".join(t.get_text() for t in bottom.get_legend().get_texts())
    assert "red scout 6" in labels and "blue scout 1" in labels


def test_teams_differ_by_more_than_color() -> None:
    blue, red = TEAM_STYLE["blue"], TEAM_STYLE["red"]
    assert (blue.color, red.color, GROUND) == ("#B8C7D9", "#F2A33A", "#0E1A2B")
    assert blue.line != red.line and blue.marker != red.marker


def test_plot_writes_a_png(demo: tuple[Replay, SegmentStore], tmp_path: Path) -> None:
    out = tmp_path / "x.png"
    plot(demo[0], out)
    blob = out.read_bytes()
    assert blob.startswith(b"\x89PNG\r\n\x1a\n") and len(blob) > 10_000
    again = tmp_path / "y.png"
    plot(demo[0], again)
    assert again.read_bytes() == blob


@pytest.mark.parametrize("duration", [1.0, 172_800.0])
def test_plot_handles_a_run_with_no_ships_or_rocks_in_the_middle(
    tmp_path: Path, duration: float
) -> None:
    data = load(SCENARIO).model_dump()
    data["blue"]["groups"] = []
    data["red"]["groups"] = []
    data["map"]["mid_clusters"] = 0
    data["duration"] = duration
    scenario = Scenario.model_validate(data)
    replay = _save(tmp_path / "e.swr", run_demo(scenario, 1), scenario, 1)
    plot(replay, tmp_path / "e.png")
    assert (tmp_path / "e.png").stat().st_size > 0
    text = " ".join(t.get_text() for t in build_figure(replay).axes[1].texts)
    assert "No enemy ship" in text
