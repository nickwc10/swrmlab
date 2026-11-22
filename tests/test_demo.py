"""The scripted crossing demo: ships arrive, within fuel, knowing only public places."""

from pathlib import Path
from typing import Any

import numpy as np
import pytest

from swarmsim.core.engine import World
from swarmsim.core.rng import Stream, uniform
from swarmsim.demo import aim_point, plan_crossing, run_demo
from swarmsim.replay import write
from swarmsim.scenario import Scenario, load, maps

SCENARIO = Path(__file__).parent.parent / "scenarios" / "two_outposts.yaml"


@pytest.fixture(scope="module")
def scenario() -> Scenario:
    return load(SCENARIO)


def _changed(scenario: Scenario, **changes: Any) -> Scenario:
    data = scenario.model_dump()
    for path, value in changes.items():
        node: Any = data
        *parents, last = path.split("__")
        for key in parents:
            node = node[int(key)] if key.isdigit() else node[key]
        node[last] = value
    return Scenario.model_validate(data)


def _ships(world: World, team: str) -> list[int]:
    """Unit ids of a team's ships; its first unit is the outpost."""
    return [u[0] for u in world.units if u[1] == team][1:]


def _burns(world: World, team: str) -> list[tuple[int, int, list[float]]]:
    ids = _ships(world, team)
    return [(e.unit, e.t_us, e.data["acc"]) for e in world.events if e.type == "burn"
            and e.unit in ids]  # fmt: skip


def _cutoff(world: World, unit: int) -> int:
    return max(e.t_us for e in world.events if e.type == "burn" and e.unit == unit)


@pytest.mark.parametrize("seed", [1, 2, 3, 4, 5])
def test_every_ship_arrives_near_rest_with_fuel_left(scenario: Scenario, seed: int) -> None:
    world = run_demo(scenario, seed)
    assert world.now_us == world.horizon_us == 172_800_000_000
    assert [e.type for e in world.events if e.type != "burn"] == []
    for team, stream, x in (
        ("blue", Stream.POLICY_BLUE, 1.5e8),
        ("red", Stream.POLICY_RED, -1.5e8),
    ):
        ids = _ships(world, team)
        assert len(ids) == 4
        for slot, unit in enumerate(ids):
            t = _cutoff(world, unit)
            assert 3600e6 < t < world.horizon_us
            pos, vel = world.store.state(unit, t)
            aim = aim_point(scenario, seed, stream, slot, len(ids), x)
            assert float(np.linalg.norm(pos - aim)) < 1.0
            assert float(np.linalg.norm(vel)) < 1e-3
            cls = scenario.ship_classes[world.units[unit][2]]
            assert world.mass(unit, world.now_us) > cls.dry_mass * 1.01


def test_aim_points_are_apart_and_near_the_enemy_home_center(scenario: Scenario) -> None:
    center = np.array([1.5e8, 0.0, 0.0])
    aims = [aim_point(scenario, 1, Stream.POLICY_BLUE, k, 4, 1.5e8) for k in range(4)]
    for i, a in enumerate(aims):
        assert a[0] == 1.5e8
        assert 2e6 < float(np.linalg.norm(a - center)) <= 4e6
        for b in aims[:i]:
            assert float(np.linalg.norm(a - b)) > 1e6
    other = aim_point(scenario, 2, Stream.POLICY_BLUE, 0, 4, 1.5e8)
    assert not np.array_equal(other, aims[0])


def test_ships_accelerate_coast_flip_and_brake(scenario: Scenario) -> None:
    world = run_demo(scenario, 1)
    coasts = {}
    for unit in _ships(world, "blue"):
        burns = [(e.t_us, e.data["acc"]) for e in world.events if e.unit == unit]
        t = [b[0] for b in burns[-4:]]
        a = [np.array(b[1]) for b in burns[-4:]]
        assert a[0][0] > 0.5 and a[2][0] < -0.5  # toward the enemy, then flipped
        assert not a[1].any() and not a[3].any()  # coast, then engine off
        assert t[0] < t[1] < t[2] < t[3]
        coasts[world.units[unit][2]] = (t[2] - t[1]) / (t[3] - t[0])
    assert coasts["attacker"] > coasts["scout"] > 0.0


def test_planning_never_uses_the_enemy_hidden_rock(
    scenario: Scenario, monkeypatch: pytest.MonkeyPatch
) -> None:
    before = run_demo(scenario, 1)

    def moved(seed: int, stream: Stream, index: int, shape: tuple[int, ...]) -> Any:
        u = uniform(seed, stream, index, shape)
        east_pick = stream == Stream.SPAWN and index == 1
        return (u + 0.5) % 1.0 if east_pick else u

    monkeypatch.setattr(maps, "uniform", moved)
    after = run_demo(scenario, 1)
    red_outpost = len(_ships(before, "blue")) + 1
    assert before.units[red_outpost][1:3] == ("red", "outpost")
    a, b = before.store.state(red_outpost, 0)[0], after.store.state(red_outpost, 0)[0]
    assert float(np.linalg.norm(a - b)) > 1e4  # red really hid elsewhere
    assert _burns(before, "blue") == _burns(after, "blue")
    assert _burns(before, "red") != _burns(after, "red")


def test_launches_from_behind_the_home_rock_go_around_it(scenario: Scenario) -> None:
    # Some start positions have the team's own rock between them and the enemy.
    detours = 0
    for seed in range(1, 11):
        world = run_demo(scenario, seed)
        assert all(u[4] is None for u in world.units), seed
        for team in ("blue", "red"):
            for unit in _ships(world, team):
                detours += sum(e.unit == unit for e in world.events) > 4
    assert detours >= 5


def test_same_seed_gives_identical_files(scenario: Scenario, tmp_path: Path) -> None:
    blobs = []
    for i, seed in enumerate((7, 7, 8)):
        path = tmp_path / f"{i}.swr"
        write(path, run_demo(scenario, seed), scenario, seed)
        blobs.append(path.read_bytes())
    assert blobs[0] == blobs[1] != blobs[2]


def test_ships_with_a_starting_speed_still_stop_at_the_target(scenario: Scenario) -> None:
    fast = _changed(
        scenario, blue__groups__0__initial_speed=3000.0, red__groups__1__initial_speed=2000.0
    )
    world = run_demo(fast, 1)
    for team in ("blue", "red"):
        for unit in _ships(world, team):
            _, vel = world.store.state(unit, _cutoff(world, unit))
            assert float(np.linalg.norm(vel)) < 1e-3


@pytest.mark.parametrize(
    "changes",
    [
        {"ship_classes__scout__max_acc": 0.0},
        {"ship_classes__scout__fuel_mass": 0.0},
        {"ship_classes__scout__fuel_mass": 1.0},  # far too little to cross in time
        {"duration": 600.0},
        {"blue__groups__0__initial_speed": 2e6},  # cannot brake from this
    ],
)
def test_a_ship_that_cannot_cross_coasts(scenario: Scenario, changes: dict[str, Any]) -> None:
    changed = _changed(scenario, **changes)
    world = run_demo(changed, 1)
    blue_scout = 1
    assert world.units[blue_scout][1:3] == ("blue", "scout")
    assert [e for e in world.events if e.unit == blue_scout and e.type == "burn"] == []


def test_a_low_fuel_ship_flies_slower_within_its_budget(scenario: Scenario) -> None:
    lean = _changed(scenario, ship_classes__attacker__fuel_mass=1000.0)  # 20.8 km/s
    world = run_demo(lean, 1)
    assert not [e for e in world.events if e.type == "fuel_out"]
    attacker = 3
    assert world.units[attacker][2] == "attacker"
    _, vel = world.store.state(attacker, _cutoff(world, attacker))
    assert float(np.linalg.norm(vel)) < 1e-3
    assert _cutoff(world, attacker) > _cutoff(run_demo(scenario, 1), attacker)


def test_teams_without_ships_are_fine(scenario: Scenario) -> None:
    empty = _changed(scenario, blue__groups=[], red__groups=[])
    assert run_demo(empty, 1).events == []


def test_plan_crossing_respects_its_limits() -> None:
    plan = plan_crossing(
        np.zeros(3), np.zeros(3), np.array([3e8, 0.0, 0.0]), 0, 172_800_000_000, 1.0, 40e3, 0.5
    )
    assert plan is not None
    (t0, a1), (t1, coast), (t2, a2), (t3, off) = plan
    assert t0 == 0 and not coast.any() and not off.any()
    speeds = [float(np.linalg.norm(a1)), float(np.linalg.norm(a2))]
    assert max(speeds) <= 1.0
    assert speeds[0] * (t1 - t0) / 1e6 + speeds[1] * (t3 - t2) / 1e6 <= 40e3
    assert (t2 - t1) / t3 == pytest.approx(0.5, abs=1e-6)
