from pathlib import Path

import numpy as np
import pytest

from swarmsim.core.engine import World
from swarmsim.scenario.build import build_world
from swarmsim.scenario.maps import build_map
from swarmsim.scenario.schema import Scenario, content_hash, load

EXAMPLE = Path(__file__).parents[1] / "scenarios" / "two_outposts.yaml"

SMALL_MAP = {
    "separation": 1e8,
    "cluster_radius": 1e6,
    "rock_radius_max": 2e4,
    "home_rocks": 20,
    "hide_candidates": 10,
    "mid_clusters": 1,
    "mid_rocks": 10,
}


def scenario(**over: object) -> Scenario:
    team = {
        "outpost": {
            "dry_mass": 1e6,
            "fuel_mass": 0.0,
            "max_acc": 0.0,
            "ve": 3000.0,
            "radius": 50.0,
        },
        "groups": [
            {"ship_class": "scout", "count": 2, "initial_speed": 100.0},
            {"ship_class": "attacker", "count": 2, "initial_speed": 50.0},
        ],
    }
    classes = {
        "scout": {"dry_mass": 100.0, "fuel_mass": 500.0, "max_acc": 1.0, "ve": 3e4, "radius": 5.0},
        "attacker": {
            "dry_mass": 500.0,
            "fuel_mass": 1500.0,
            "max_acc": 1.0,
            "ve": 3e4,
            "radius": 10.0,
        },
    }
    d: dict[str, object] = {
        "name": "t",
        "duration": 3600.0,
        "map": SMALL_MAP,
        "ship_classes": classes,
        "blue": team,
        "red": team,
    }
    d.update(over)
    return Scenario.model_validate(d)


def state(w: World, u: int) -> tuple[np.ndarray, np.ndarray]:
    p, v = w.store.state(u, 0)
    return np.array(p), np.array(v)


def test_units_are_spawned_in_fixed_order_by_team() -> None:
    w = build_world(scenario(), 3)
    assert [(u[1], u[2]) for u in w.units] == (
        [("blue", "outpost"), ("blue", "scout"), ("blue", "scout")]
        + [("blue", "attacker"), ("blue", "attacker")]
        + [("red", "outpost"), ("red", "scout"), ("red", "scout")]
        + [("red", "attacker"), ("red", "attacker")]
    )
    assert w.horizon_us == 3_600_000_000
    assert w.now_us == 0
    assert all(u[3] == 0 and u[4] is None for u in w.units)


def test_the_world_uses_the_scenario_light_speed_and_map() -> None:
    s = scenario(c=1e8)
    w = build_world(s, 3)
    assert w.c == 1e8
    assert w.rocks.centers.shape[0] == 2 * (20 + 10)


def test_outposts_sit_just_outside_their_hiding_rock_and_touch_nothing() -> None:
    s = scenario()
    tol = s.map.outpost_clearance * (1 + 1e-9)
    for swap in (False, True):
        for seed in range(10):
            w = build_world(s, seed, swap)
            m = build_map(s.map, seed, swap)
            for first, hide in ((0, m.blue_hide), (5, m.red_hide)):
                p, v = state(w, first)
                assert not v.any()
                d = np.linalg.norm(w.rocks.centers - p, axis=1) - w.rocks.radii - 50.0
                assert int(np.argmin(d)) == hide, (seed, swap, first)
                assert 0.0 < d[hide] <= tol
                assert (d > 0).all()
                assert np.sum(d <= tol) == 1  # every other rock is farther than the clearance


def test_west_and_east_placements_differ_and_are_not_mirrors() -> None:
    s = scenario()
    for seed in range(10):
        w = build_world(s, seed)
        m = build_map(s.map, seed)
        for k in range(5):  # the outpost, then each ship
            blue = state(w, k)[0] - w.rocks.centers[m.blue_hide]
            red = state(w, 5 + k)[0] - w.rocks.centers[m.red_hide]
            assert not np.allclose(blue, red, rtol=0, atol=1e-6), (seed, k)
            assert not np.allclose(blue * np.array([-1.0, 1.0, 1.0]), red, rtol=0, atol=1e-6)


def test_outposts_are_inert() -> None:
    w = build_world(scenario(), 1)
    for u in (0, 5):
        assert w.mass(u, 0) == 1e6
        with pytest.raises(ValueError, match="limit"):
            w.command_acc(u, 0, (1e-3, 0.0, 0.0))


def test_ships_start_near_their_outpost_clear_of_rocks_and_moving_at_the_enemy() -> None:
    s = scenario()
    clearance = s.map.outpost_clearance
    for seed in range(10):
        w = build_world(s, seed)
        for first in (0, 5):
            outpost, _ = state(w, first)
            sign = 1.0 if outpost[0] < 0 else -1.0
            for k, speed in zip(range(1, 5), (100.0, 100.0, 50.0, 50.0), strict=True):
                p, v = state(w, first + k)
                radius = 5.0 if k <= 2 else 10.0
                assert np.allclose(v, (sign * speed, 0.0, 0.0))
                dist = float(np.linalg.norm(p - outpost))
                assert dist >= 50.0 + radius + clearance
                assert dist <= 50.0 + radius + clearance + s.map.ship_spread + 1e-6
                gap = np.linalg.norm(w.rocks.centers - p, axis=1) - w.rocks.radii - radius
                assert gap.min() > clearance


def test_blue_is_west_unless_swapped() -> None:
    w = build_world(scenario(), 2)
    assert state(w, 0)[0][0] < 0 < state(w, 5)[0][0]
    ws = build_world(scenario(), 2, swap=True)
    assert state(ws, 0)[0][0] > 0 > state(ws, 5)[0][0]


def test_swap_exchanges_the_teams_positions_exactly() -> None:
    s = scenario()  # both teams are identical, so only the side differs
    for seed in range(8):
        a, b = build_world(s, seed), build_world(s, seed, swap=True)
        assert [u[1:3] for u in a.units] == [u[1:3] for u in b.units]  # ids depend on team
        for k in range(5):
            pa, va = state(a, k)
            pb, vb = state(b, 5 + k)
            assert np.array_equal(pa, pb), (seed, k)
            assert np.array_equal(va, vb)
            qa, wa = state(a, 5 + k)
            qb, wb = state(b, k)
            assert np.array_equal(qa, qb)
            assert np.array_equal(wa, wb)
        assert np.array_equal(a.rocks.centers, b.rocks.centers)


def test_same_seed_same_world_and_different_seeds_differ() -> None:
    s = scenario()
    a, b, c = build_world(s, 4), build_world(s, 4), build_world(s, 5)
    assert a.store.table().tobytes() == b.store.table().tobytes()
    assert a.store.table().tobytes() != c.store.table().tobytes()
    assert not np.array_equal(a.rocks.centers, c.rocks.centers)


def test_a_built_world_carries_the_settings_that_made_it() -> None:
    s = scenario()
    for seed, swap in ((4, False), (4, True), (2**63 - 1, False)):
        assert build_world(s, seed, swap).provenance == (content_hash(s), seed, swap)
    stamp = build_world(s, np.int64(4)).provenance  # type: ignore[arg-type]
    assert stamp is not None
    assert [type(x) for x in stamp] == [str, int, bool]


def test_a_built_world_runs() -> None:
    w = build_world(scenario(), 1)
    w.run_until(w.horizon_us)
    assert w.now_us == w.horizon_us


def test_unflyable_ship_classes_fail_at_load_naming_group_and_class() -> None:
    with pytest.raises(ValueError, match=r"blue\.groups\[0\].*scout.*top speed"):
        scenario(c=1e5)  # top speed 0.1 c = 1e4 m/s is below the classes' delta-v


def test_unplaceable_outposts_raise_naming_the_setting() -> None:
    crowded = {**SMALL_MAP, "home_rocks": 500, "outpost_clearance": 1e6}  # rocks fill the space
    with pytest.raises(ValueError, match="outpost_clearance"):
        build_world(scenario(map=crowded), 0)


def test_unplaceable_ships_raise_naming_the_setting(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("swarmsim.scenario.build.ATTEMPTS", 1)
    # With one try per ship, some ship in some seed lands inside a rock or too close to one.
    messages = set()
    for seed in range(200):
        try:
            build_world(scenario(map={**SMALL_MAP, "ship_spread": 3e5}), seed)
        except ValueError as err:
            messages.add(str(err))
    assert messages
    assert all("ship_spread" in m or "outpost_clearance" in m for m in messages)


def test_example_scenario_builds_and_runs_deterministically() -> None:
    s = load(EXAMPLE)
    w = build_world(s, 7)
    assert len(w.units) == 10
    assert w.horizon_us == 172_800_000_000
    assert w.rocks.centers.shape[0] == 2 * (60 + 2 * 30)
    p0, _ = state(w, 0)
    p5, _ = state(w, 5)
    assert abs(abs(p0[0]) - 1.5e8) < 3e6 and abs(abs(p5[0]) - 1.5e8) < 3e6
    assert build_world(s, 7).store.table().tobytes() == w.store.table().tobytes()
    assert len(content_hash(s)) == 64
