import dataclasses
import json
import math
from collections.abc import Sequence
from typing import Any

import numpy as np
import pytest

from swarmsim.core import engine
from swarmsim.core.engine import Event, World
from swarmsim.core.geometry import Rocks, first_within
from swarmsim.core.light import C
from swarmsim.core.segments import SegmentStore

S = 1_000_000  # microseconds per second
HOUR = 3600 * S
ZERO = (0.0, 0.0, 0.0)

DEFAULTS: dict[str, Any] = {
    "team": "blue",
    "cls": "scout",
    "dry_mass": 100.0,
    "fuel_mass": 200.0,
    "max_acc": 10.0,
    "ve": 3000.0,
    "radius": 10.0,
}


def rocks(*spheres: tuple[float, float, float, float], cluster: Sequence[int] = ()) -> Rocks:
    """Rocks from (x, y, z, radius); each is its own cluster unless told otherwise."""
    arr = np.array(spheres, dtype=np.float64).reshape(-1, 4)
    ids = np.array(cluster or range(len(spheres)), dtype=np.int64)
    return Rocks(arr[:, :3], arr[:, 3], ids)


def make_world(field: Rocks | None = None, horizon_us: int = 10 * HOUR, c: float = C) -> World:
    return World(SegmentStore(), rocks() if field is None else field, c, horizon_us=horizon_us)


def spawn(
    world: World,
    pos: Sequence[float] = ZERO,
    vel: Sequence[float] = ZERO,
    t_us: int = 0,
    **over: Any,
) -> int:
    return world.spawn(t_us, pos, vel, **{**DEFAULTS, **over})


def kinds(world: World, unit: int | None = None) -> list[str]:
    return [e.type for e in world.events if unit is None or e.unit == unit]


def dump(world: World) -> tuple[list[Event], bytes, Any]:
    return list(world.events), world.store.table().tobytes(), world.units


# --- construction and time ------------------------------------------------------------


def test_run_until_advances_now_and_bounds_time() -> None:
    w = make_world(horizon_us=100)
    assert w.now_us == 0
    w.run_until(40)
    assert w.now_us == 40
    w.run_until(40)
    with pytest.raises(ValueError):
        w.run_until(39)
    with pytest.raises(ValueError):
        w.run_until(101)
    assert w.now_us == 40
    w.run_until(100)
    assert w.now_us == 100


def test_world_rejects_bad_arguments() -> None:
    with pytest.raises(ValueError):
        make_world(horizon_us=-1)
    for c in (0.0, -1.0, math.inf, math.nan):
        with pytest.raises(ValueError):
            make_world(c=c)
    store = SegmentStore()
    store.add_unit(0, ZERO, ZERO, ZERO)
    with pytest.raises(ValueError):
        World(store, rocks(), horizon_us=10)


def test_light_speed_defaults_to_c() -> None:
    assert World(SegmentStore(), rocks(), horizon_us=10).c == C


def test_event_is_frozen_with_json_safe_data() -> None:
    w = make_world(rocks((1000.0, 0.0, 0.0, 50.0)))
    u = spawn(w, vel=(1.0, 0.0, 0.0), fuel_mass=1.0)
    w.command_acc(np.int64(u), np.int64(5), np.array([10.0, 0.0, 0.0]))
    w.run_until(w.horizon_us)
    assert kinds(w) == ["burn", "fuel_out", "collision", "end"]
    for e in w.events:
        assert type(e.data) is dict
        assert json.loads(json.dumps(e.data)) == e.data
        assert type(e.t_us) is int and type(e.unit) is int
        assert e.other is None
    with pytest.raises(dataclasses.FrozenInstanceError):
        w.events[0].t_us = 3  # type: ignore[misc]


# --- spawn ----------------------------------------------------------------------------


def test_spawn_takes_effect_immediately_with_zero_acceleration() -> None:
    w = make_world()
    u = spawn(w, (1.0, 2.0, 3.0), (4.0, 5.0, 6.0), team="red", cls="attacker")
    assert w.units == [(u, "red", "attacker", 0, None)]
    assert w.mass(u, 0) == 300.0
    pos, vel = w.store.state(u, 2 * S)
    assert pos.tolist() == [9.0, 12.0, 15.0]
    assert vel.tolist() == [4.0, 5.0, 6.0]
    assert w.events == []


def test_units_are_listed_in_id_order_with_birth_and_end() -> None:
    w = make_world(rocks((1000.0, 0.0, 0.0, 50.0)))
    a = spawn(w, vel=(94.0, 0.0, 0.0))
    w.run_until(S)
    b = spawn(w, (0.0, 500.0, 0.0), t_us=S, team="red", cls="attacker")
    w.run_until(20 * S)
    assert w.units == [(a, "blue", "scout", 0, 10 * S), (b, "red", "attacker", S, None)]


@pytest.mark.parametrize(
    "over",
    [
        {"dry_mass": 0.0},
        {"dry_mass": -1.0},
        {"ve": 0.0},
        {"ve": -1.0},
        {"fuel_mass": -1.0},
        {"max_acc": -1.0},
        {"radius": -1.0},
        {"pos": (math.nan, 0.0, 0.0)},
        {"vel": (0.0, math.inf, 0.0)},
        {"pos": (0.0, 0.0)},
        *({k: v} for k in ("dry_mass", "fuel_mass", "max_acc", "ve", "radius")
          for v in (math.nan, math.inf)),
    ],
)  # fmt: skip
def test_spawn_rejects_bad_values_and_adds_nothing(over: dict[str, Any]) -> None:
    w = make_world()
    with pytest.raises(ValueError):
        spawn(w, **over)
    assert len(w.store) == 0
    assert w.units == []


def test_spawn_allows_zero_fuel_acceleration_and_radius() -> None:
    w = make_world()
    u = spawn(w, fuel_mass=0.0, max_acc=0.0, radius=0.0)
    assert w.mass(u, 0) == 100.0


def test_spawn_time_bounds() -> None:
    w = make_world(horizon_us=100)
    w.run_until(50)
    for t in (49, 101):
        with pytest.raises(ValueError):
            spawn(w, t_us=t)
    assert spawn(w, t_us=50) == 0
    assert spawn(w, t_us=100) == 1


def test_spawn_touching_a_rock_is_rejected() -> None:
    field = rocks((1000.0, 0.0, 0.0, 100.0))
    for x in (1000.0, 890.0, 889.9999):
        w = make_world(field)
        with pytest.raises(ValueError):
            spawn(w, (x, 0.0, 0.0))
        with pytest.raises(ValueError):
            spawn(w, (x, 0.0, 0.0), t_us=w.horizon_us)
        assert len(w.store) == 0
    assert spawn(make_world(field), (889.0, 0.0, 0.0)) == 0


def test_spawn_speed_budget_is_a_tenth_of_c() -> None:
    c = 1e5
    dv = 3000.0 * math.log(3.0)
    w = make_world(c=c)
    spawn(w, vel=(0.1 * c - dv - 1e-3, 0.0, 0.0))
    with pytest.raises(ValueError):
        spawn(w, vel=(0.0, -(0.1 * c - dv + 1e-3), 0.0))


def test_future_spawn_is_born_at_its_time() -> None:
    w = make_world(rocks((1000.0, 0.0, 0.0, 50.0)))
    u = spawn(w, vel=(94.0, 0.0, 0.0), t_us=5 * S)
    assert w.units == [(u, "blue", "scout", 5 * S, None)]
    with pytest.raises(ValueError):
        w.mass(u, 0)
    with pytest.raises(ValueError):
        w.command_acc(u, 5 * S - 1, ZERO)
    w.command_acc(u, 5 * S, (0.0, 0.0, 0.0))
    w.run_until(20 * S)
    assert [(e.t_us, e.type) for e in w.events] == [
        (5 * S, "burn"),
        (15 * S, "collision"),
        (15 * S, "end"),
    ]
    assert w.mass(u, 5 * S) == 300.0


# --- fuel -----------------------------------------------------------------------------


def test_total_delta_v_matches_the_rocket_equation() -> None:
    w = make_world()
    u = spawn(w)
    w.command_acc(u, 10 * S, (7.0, 0.0, 0.0))
    w.command_acc(u, 60 * S, ZERO)
    w.command_acc(u, 100 * S, (0.6, 0.8, 0.0))
    w.command_acc(u, 300 * S, (3.0, 0.0, 4.0))
    w.run_until(w.horizon_us)
    segs = w.store.table()
    dv = 0.0
    for i in range(len(segs) - 1):
        a = float(np.linalg.norm(segs["acc"][i]))
        dv += a * int(segs["t0"][i + 1] - segs["t0"][i]) / 1e6
    assert segs["acc"][-1].tolist() == [0.0, 0.0, 0.0]
    assert dv == pytest.approx(3000.0 * math.log(3.0), rel=1e-6)
    assert kinds(w) == ["burn", "burn", "burn", "burn", "fuel_out"]


@pytest.mark.parametrize("acc", [0.37, 1.0, 9.81])
def test_fuel_out_time_rounds_down_and_stops_the_burn(acc: float) -> None:
    w = make_world()
    u = spawn(w)
    w.command_acc(u, 1234, (0.0, acc, 0.0))
    w.run_until(w.horizon_us)
    exact = 1234 + 1e6 * 3000.0 * math.log(3.0) / acc
    (e,) = (e for e in w.events if e.type == "fuel_out")
    assert e.unit == u
    assert exact - 1.0 - 1e-3 < e.t_us <= exact + 1e-3
    segs = w.store.table()
    assert int(segs["t0"][-1]) == e.t_us
    assert segs["acc"][-1].tolist() == [0.0, 0.0, 0.0]
    assert w.mass(u, e.t_us) == 100.0
    assert 100.0 < w.mass(u, e.t_us - 1) < 100.001
    assert w.mass(u, w.now_us) == 100.0


def test_mass_follows_burns_and_is_continuous() -> None:
    w = make_world()
    u = spawn(w)
    w.command_acc(u, 10 * S, (3.0, 0.0, 4.0))
    w.command_acc(u, 30 * S, ZERO)
    w.command_acc(u, 50 * S, (0.0, 2.0, 0.0))
    w.run_until(60 * S)
    assert w.mass(u, 0) == 300.0
    assert w.mass(u, 10 * S) == 300.0
    m20 = 300.0 * math.exp(-5.0 * 10.0 / 3000.0)
    m30 = 300.0 * math.exp(-5.0 * 20.0 / 3000.0)
    assert w.mass(u, 20 * S) == pytest.approx(m20, rel=1e-12)
    assert w.mass(u, 30 * S) == pytest.approx(m30, rel=1e-12)
    assert w.mass(u, 50 * S) == w.mass(u, 30 * S)
    assert w.mass(u, 60 * S) == pytest.approx(m30 * math.exp(-20.0 / 3000.0), rel=1e-12)


def test_mass_time_bounds() -> None:
    w = make_world(rocks((1000.0, 0.0, 0.0, 50.0)))
    u = spawn(w, vel=(94.0, 0.0, 0.0))
    w.run_until(S)
    w.mass(u, S)
    for t in (-1, S + 1):
        with pytest.raises(ValueError):
            w.mass(u, t)
    with pytest.raises(ValueError):
        w.mass(u + 1, S)
    w.run_until(20 * S)
    w.mass(u, 10 * S)
    with pytest.raises(ValueError):
        w.mass(u, 10 * S + 1)


def test_burn_without_fuel_does_nothing_and_logs_no_second_fuel_out() -> None:
    w = make_world()
    u = spawn(w, fuel_mass=1.0)
    w.command_acc(u, 0, (10.0, 0.0, 0.0))
    w.run_until(HOUR)
    assert kinds(w) == ["burn", "fuel_out"]
    _, before = w.store.state(u, HOUR)
    w.command_acc(u, HOUR, (10.0, 0.0, 0.0))
    w.run_until(2 * HOUR)
    assert kinds(w) == ["burn", "fuel_out", "burn"]
    assert w.events[-1].data == {"acc": [0.0, 0.0, 0.0]}
    _, after = w.store.state(u, 2 * HOUR)
    assert after.tolist() == before.tolist()
    assert w.mass(u, 2 * HOUR) == 100.0


def test_unit_born_without_fuel_never_burns_or_runs_out() -> None:
    w = make_world()
    u = spawn(w, fuel_mass=0.0)
    w.command_acc(u, 0, (1.0, 0.0, 0.0))
    w.run_until(HOUR)
    assert kinds(w) == ["burn"]
    assert w.store.state(u, HOUR)[1].tolist() == [0.0, 0.0, 0.0]


def test_replaced_burn_drops_the_old_fuel_out() -> None:
    w = make_world()
    u = spawn(w)
    w.command_acc(u, 0, (10.0, 0.0, 0.0))  # would run dry near 329.6 s
    w.command_acc(u, 100 * S, ZERO)
    w.run_until(HOUR)
    assert kinds(w) == ["burn", "burn"]
    assert w.mass(u, HOUR) > 100.0


def test_fuel_out_beyond_horizon_is_not_logged_and_at_horizon_is() -> None:
    probe = make_world()
    u = spawn(probe)
    probe.command_acc(u, 0, (10.0, 0.0, 0.0))
    probe.run_until(probe.horizon_us)
    t_out = probe.events[-1].t_us
    for horizon, expected in ((t_out - 1, ["burn"]), (t_out, ["burn", "fuel_out"])):
        w = make_world(horizon_us=horizon)
        spawn(w)
        w.command_acc(0, 0, (10.0, 0.0, 0.0))
        w.run_until(horizon)
        assert kinds(w) == expected


def test_tiny_acceleration_never_reaches_fuel_out() -> None:
    w = make_world()
    u = spawn(w)
    w.command_acc(u, 0, (5e-324, 0.0, 0.0))
    w.run_until(w.horizon_us)
    assert kinds(w) == ["burn"]


def test_rounding_never_takes_mass_below_dry() -> None:
    # A burn so slow that one microsecond changes mass by less than one ulp; found by search.
    dry, mass = 1.7989584595845867, 1.8657758314204702
    acc, t_out = 5.475892881946525e-09, 19_979_809_091_211_068
    w = make_world(horizon_us=t_out + HOUR)
    u = spawn(w, dry_mass=dry, fuel_mass=mass - dry)
    w.command_acc(u, 0, (acc, 0.0, 0.0))
    w.command_acc(u, t_out - 1, (2.0 * acc, 0.0, 0.0))
    w.run_until(w.horizon_us)
    assert kinds(w) == ["burn", "burn", "fuel_out"]
    assert w.mass(u, t_out - 1) == dry


# --- commands -------------------------------------------------------------------------


def test_acceleration_limit_is_checked_at_the_call_with_relative_slack() -> None:
    w = make_world()
    u = spawn(w, max_acc=5.0)
    w.command_acc(u, 0, (3.0, 0.0, 4.0))
    w.command_acc(u, 1, (5.0 * (1 + 5e-13), 0.0, 0.0))
    for bad in ((5.0 * (1 + 3e-12), 0.0, 0.0), (3.0, 1.0, 4.0), (1e200, 0.0, 0.0)):
        with pytest.raises(ValueError):
            w.command_acc(u, 2, bad)
    with pytest.raises(ValueError):
        w.command_acc(u, 2, (math.nan, 0.0, 0.0))
    w.run_until(10)
    assert [e.t_us for e in w.events] == [0, 1]


def test_command_bounds() -> None:
    w = make_world(rocks((1000.0, 0.0, 0.0, 50.0)), horizon_us=20 * S)
    u = spawn(w, vel=(94.0, 0.0, 0.0))
    w.run_until(S)
    with pytest.raises(ValueError):
        w.command_acc(u, S - 1, ZERO)
    with pytest.raises(ValueError):
        w.command_acc(u, 20 * S + 1, ZERO)
    with pytest.raises(ValueError):
        w.command_acc(u + 1, S, ZERO)
    with pytest.raises(ValueError):
        w.command_acc(-1, S, ZERO)
    w.command_acc(u, 20 * S, ZERO)
    w.run_until(11 * S)
    with pytest.raises(ValueError):
        w.command_acc(u, 12 * S, ZERO)


def test_command_at_now_after_run_until_now_is_applied_by_another_run() -> None:
    w = make_world()
    u = spawn(w)
    w.run_until(10 * S)
    w.command_acc(u, 10 * S, (1.0, 0.0, 0.0))
    assert w.events == []
    w.run_until(10 * S)
    assert [(e.t_us, e.type, e.unit, e.data) for e in w.events] == [
        (10 * S, "burn", u, {"acc": [1.0, 0.0, 0.0]})
    ]
    assert w.store.state(u, 12 * S)[1].tolist() == [2.0, 0.0, 0.0]


def test_last_queued_command_for_a_unit_and_instant_wins() -> None:
    w = make_world()
    a = spawn(w)
    b = spawn(w, (0.0, 1e6, 0.0))
    w.command_acc(a, S, (1.0, 0.0, 0.0))
    w.command_acc(b, S, (3.0, 0.0, 0.0))
    w.command_acc(a, S, (2.0, 0.0, 0.0))  # replaces the first, and goes after b
    w.command_acc(a, 2 * S, ZERO)
    w.run_until(3 * S)
    assert [(e.t_us, e.unit, e.data["acc"][0]) for e in w.events] == [
        (S, b, 3.0),
        (S, a, 2.0),
        (2 * S, a, 0.0),
    ]


def test_command_already_applied_at_an_instant_stays_logged() -> None:
    w = make_world()
    u = spawn(w)
    for ax in (1.0, 2.0):
        w.command_acc(u, 0, (ax, 0.0, 0.0))
        w.run_until(0)
    assert [e.data["acc"][0] for e in w.events] == [1.0, 2.0]
    w.run_until(S)
    assert w.store.state(u, S)[1].tolist() == [2.0, 0.0, 0.0]


def test_rejected_replacement_keeps_the_queued_command() -> None:
    w = make_world()
    u = spawn(w)
    w.command_acc(u, S, (1.0, 0.0, 0.0))
    with pytest.raises(ValueError):
        w.command_acc(u, S, (1e3, 0.0, 0.0))
    w.run_until(S)
    assert w.events[0].data == {"acc": [1.0, 0.0, 0.0]}


# --- collisions -----------------------------------------------------------------------


def test_ship_crossing_a_rock_at_55_km_s_collides() -> None:
    w = make_world(rocks((1e6, 0.0, 0.0, 100.0), (1e9, 0.0, 0.0, 100.0)), horizon_us=12 * HOUR)
    u = spawn(w, (1e6 + 200.0, 0.0, 0.0), (55e3, 0.0, 0.0))
    w.run_until(w.horizon_us)
    exact = 1e6 * (1e9 - 110.0 - 1e6 - 200.0) / 55e3
    assert [(e.type, e.unit, e.other) for e in w.events] == [
        ("collision", u, None),
        ("end", u, None),
    ]
    assert w.events[0].data["rock"] == 1
    assert w.events[0].t_us == w.events[1].t_us
    assert abs(w.events[0].t_us - exact) <= 1.0
    assert w.units[0][4] == w.events[0].t_us


def test_contact_time_rounds_up() -> None:
    w = make_world(rocks((1060.0, 0.0, 0.0, 50.0)))
    spawn(w, vel=(3.0, 0.0, 0.0))  # gap 1000 m at 3 m/s: 333333333.33 us
    w.run_until(w.horizon_us)
    assert w.events[0].t_us == 333_333_334


def test_units_touching_each_other_do_not_collide() -> None:
    w = make_world()
    spawn(w)
    spawn(w, vel=(1.0, 0.0, 0.0))
    w.run_until(w.horizon_us)
    assert w.events == []


def test_contact_at_the_horizon_counts_and_after_it_does_not() -> None:
    field = rocks((1060.0, 0.0, 0.0, 50.0))
    for horizon, n in ((10 * S, 2), (10 * S - 1000, 0)):
        w = make_world(field, horizon_us=horizon)
        spawn(w, vel=(100.0, 0.0, 0.0))
        w.run_until(horizon)
        assert len(w.events) == n


def test_contact_rounded_past_the_horizon_is_held_at_it() -> None:
    horizon = 123  # 123 us in seconds and back rounds to just above 123
    assert math.ceil(horizon / 1e6 * 1e6) == horizon + 1
    gap = 100.0 * horizon / 1e6 + 5e-4  # within contact tolerance at the horizon
    w = make_world(rocks((gap + 60.0, 0.0, 0.0, 50.0)), horizon_us=horizon)
    spawn(w, vel=(100.0, 0.0, 0.0))
    w.run_until(horizon)
    assert [(e.t_us, e.type) for e in w.events] == [(horizon, "collision"), (horizon, "end")]


def test_stale_collision_is_dropped_when_a_unit_burns_away() -> None:
    w = make_world(rocks((10_060.0, 0.0, 0.0, 50.0)))
    u = spawn(w, vel=(100.0, 0.0, 0.0))  # would hit at 100 s
    w.command_acc(u, 50 * S, (-10.0, 0.0, 0.0))
    w.run_until(HOUR)
    assert kinds(w) == ["burn", "fuel_out"]
    assert w.units[0][4] is None


def test_collision_is_found_when_a_unit_burns_into_a_rock() -> None:
    w = make_world(rocks((0.0, 5060.0, 0.0, 50.0)))
    u = spawn(w)
    w.command_acc(u, 10 * S, (0.0, 4.0, 0.0))  # 5000 m at 4 m/s2: 50 s
    w.run_until(HOUR)
    assert kinds(w) == ["burn", "collision", "end"]
    assert abs(w.events[1].t_us - 60 * S) <= 10
    assert w.store.state(u, w.events[1].t_us)[0][1] == pytest.approx(5000.0, abs=2e-3)


def test_collision_after_fuel_out_uses_the_coasting_path() -> None:
    w = make_world(rocks((1e7, 0.0, 0.0, 50.0)))
    u = spawn(w)
    w.command_acc(u, 0, (10.0, 0.0, 0.0))
    w.run_until(w.horizon_us)
    assert kinds(w) == ["burn", "fuel_out", "collision", "end"]
    dv = 3000.0 * math.log(3.0)
    t_burn = dv / 10.0
    expected = t_burn + (1e7 - 60.0 - 0.5 * dv * t_burn) / dv
    assert abs(w.events[2].t_us - 1e6 * expected) < 1e3


def test_dead_unit_command_is_dropped_including_at_the_collision_instant() -> None:
    w = make_world(rocks((1060.0, 0.0, 0.0, 50.0)))
    u = spawn(w, vel=(100.0, 0.0, 0.0))  # hits at exactly 10 s
    w.command_acc(u, 10 * S, (-1.0, 0.0, 0.0))
    w.command_acc(u, 11 * S, (-1.0, 0.0, 0.0))
    w.run_until(HOUR)
    assert [(e.t_us, e.type) for e in w.events] == [(10 * S, "collision"), (10 * S, "end")]


def test_fuel_out_goes_before_a_command_at_the_same_instant() -> None:
    probe = make_world()
    spawn(probe)
    probe.command_acc(0, 0, (10.0, 0.0, 0.0))
    probe.run_until(probe.horizon_us)
    t_out = probe.events[-1].t_us
    w = make_world()
    u = spawn(w)
    w.command_acc(u, t_out, (0.0, 5.0, 0.0))  # lower sequence number than the fuel_out
    w.command_acc(u, 0, (10.0, 0.0, 0.0))
    w.run_until(w.horizon_us)
    assert [(e.t_us, e.type) for e in w.events] == [
        (0, "burn"),
        (t_out, "fuel_out"),
        (t_out, "burn"),
    ]
    assert w.events[-1].data == {"acc": [0.0, 0.0, 0.0]}


def test_collision_goes_before_fuel_out_at_the_same_instant() -> None:
    vel = (20e3, 0.0, 0.0)
    probe = make_world()
    spawn(probe, vel=vel)
    probe.command_acc(0, 0, (10.0, 0.0, 0.0))
    probe.run_until(probe.horizon_us)
    t_out = probe.events[-1].t_us
    pos, v = probe.store.state(0, t_out)
    touch = float(pos[0]) - 0.5e-6 * float(v[0])  # front of the unit half a us before
    w = make_world(rocks((touch + 60.0, 0.0, 0.0, 50.0)))
    u = spawn(w, vel=vel)
    w.command_acc(u, 0, (10.0, 0.0, 0.0))
    w.run_until(w.horizon_us)
    assert [(e.t_us, e.type) for e in w.events] == [
        (0, "burn"),
        (t_out, "collision"),
        (t_out, "end"),
    ]
    assert w.mass(u, t_out) > 100.0


# --- slicing --------------------------------------------------------------------------


def busy_world() -> World:
    horizon = 2 * HOUR
    field = rocks(
        (3e6, 0.0, 0.0, 500.0),
        (3e6, 2e3, 0.0, 800.0),
        (-2e6, 1e6, 0.0, 300.0),
        (0.0, -4e6, 0.0, 400.0),
        cluster=(0, 0, 1, 2),
    )
    w = make_world(field, horizon_us=horizon)
    coaster = spawn(w, vel=(500.0, 0.0, 0.0))  # hits near 5996 s
    burner = spawn(w, (0.0, 1e5, 0.0), (0.0, 0.0, 10.0))
    diver = spawn(w, (0.0, -1e6, 0.0))
    dodger = spawn(w, (-1.9e6, 1e6, 0.0), (-50.0, 0.0, 0.0), fuel_mass=20.0)
    spawn(w, (5e5, 5e5, 5e5), (1.0, 2.0, 3.0), t_us=1000 * S, team="red")
    slice_us = horizon // 100
    w.command_acc(burner, 7 * slice_us, (3.0, 1.0, 0.0))  # exactly on a slice boundary
    w.command_acc(burner, 150 * S + 333, (0.0, -2.0, 1.0))
    w.command_acc(burner, 700 * S, (0.0, 0.0, 5.0))
    w.command_acc(diver, 333 * S + 7, (0.0, -8.0, 0.0))
    w.command_acc(dodger, 900 * S, (0.0, 1.0, 0.0))
    w.command_acc(coaster, 4000 * S, (1e-3, 0.0, 0.0))
    return w


def test_one_call_and_100_calls_agree() -> None:
    one = busy_world()
    one.run_until(one.horizon_us)
    many = busy_world()
    cuts = [many.horizon_us * i // 100 for i in range(1, 101)]
    for t in cuts:
        many.run_until(t)
    assert dump(one) == dump(many)
    for u, _, _, birth, end in one.units:
        last = one.horizon_us if end is None else end
        for t in range(birth, last + 1, 97 * S):
            assert one.mass(u, t) == many.mass(u, t)
    after_first = [e.type for e in one.events if e.t_us > cuts[0]]
    assert {"burn", "fuel_out", "collision", "end"} <= set(after_first)
    assert any(e.t_us in cuts for e in one.events)


def test_slicing_at_every_event_time_agrees() -> None:
    one = busy_world()
    one.run_until(one.horizon_us)
    many = busy_world()
    for t in sorted({e.t_us + d for e in one.events for d in (-1, 0, 1)} | {one.horizon_us}):
        many.run_until(t)
    assert dump(one) == dump(many)


# --- pruning --------------------------------------------------------------------------


def unpruned_contact(
    segment: Any, field: Rocks, radius: float, dt_max: float
) -> tuple[float, int] | None:
    hits = []
    for i in range(len(field.radii)):
        dt = first_within(
            segment["pos"] - field.centers[i],
            segment["vel"],
            segment["acc"],
            radius + float(field.radii[i]),
            dt_max,
        )
        if dt is not None:
            hits.append((dt, i))
    return min(hits, default=None)


def test_pruning_matches_unpruned_checks_on_a_seeded_field() -> None:
    rng = np.random.default_rng(20261122)
    origin = np.array([3e9, -1e9, 5e8])
    n_clusters, per = 6, 8
    middles = origin + rng.uniform(-1e7, 1e7, (n_clusters, 3))
    ids = rng.permutation(np.repeat(np.arange(n_clusters) * 10 - 20, per))
    slot = (ids + 20) // 10
    centers = middles[slot] + rng.uniform(-2e5, 2e5, (n_clusters * per, 3))
    radii = rng.uniform(500.0, 2e4, n_clusters * per)
    field = Rocks(centers, radii, ids)
    horizon = 4 * HOUR
    w = World(SegmentStore(), field, horizon_us=horizon)
    unit_radius = 25.0
    for k in range(240):
        target = int(rng.integers(len(radii)))
        toward = rng.normal(size=3)
        toward /= np.linalg.norm(toward)
        side = np.cross(toward, rng.normal(size=3))
        side /= np.linalg.norm(side)
        # Aim near the rim of a rock: many grazing hits and grazing misses.
        miss = (radii[target] + unit_radius) * rng.uniform(0.9, 1.1)
        start = centers[target] - toward * 4e7 + side * miss
        try:
            u = w.spawn(
                0,
                start,
                toward * rng.uniform(1e4, 8e4),
                team="blue",
                cls="probe",
                dry_mass=100.0,
                fuel_mass=10.0,
                max_acc=10.0,
                ve=3e4,
                radius=unit_radius,
            )
        except ValueError:  # started inside some rock
            continue
        if k % 2:
            t = int(rng.integers(0, 600 * S))
            w.command_acc(u, t, rng.normal(size=3))
    w.run_until(horizon)

    collisions = {e.unit: (e.t_us, e.data["rock"]) for e in w.events if e.type == "collision"}
    table = w.store.table()
    for u, _, _, _, end in w.units:
        segs = table[table["unit"] == u]
        found = None
        for j, seg in enumerate(segs):
            t0 = int(seg["t0"])
            hit = unpruned_contact(seg, field, unit_radius, (horizon - t0) / 1e6)
            if hit is None:
                continue
            t_hit = t0 + math.ceil(hit[0] * 1e6)
            if j + 1 == len(segs) or t_hit <= int(segs["t0"][j + 1]):
                found = (t_hit, hit[1])
                break
        assert collisions.get(u) == found
        assert end == (None if found is None else found[0])
    n = len(w.units)
    assert n > 200
    assert 0.2 * n < len(collisions) < 0.9 * n
    assert len({int(ids[rock]) for _, rock in collisions.values()}) == n_clusters
    assert any(e.type == "fuel_out" for e in w.events)


def test_unreachable_clusters_are_not_checked_rock_by_rock(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = 0

    def counting(*args: Any) -> float | None:
        nonlocal calls
        calls += 1
        return first_within(*args)

    monkeypatch.setattr(engine, "first_within", counting)
    near = [(1e5 + 1e3 * i, 0.0, 0.0, 100.0) for i in range(10)]
    far = [(-1e7 - 1e3 * i, 0.0, 0.0, 100.0) for i in range(10)]
    w = make_world(rocks(*near, *far, cluster=[3] * 10 + [7] * 10))
    spawn(w, vel=(100.0, 0.0, 0.0))
    assert calls == 2 + 10
    w.run_until(w.horizon_us)
    assert calls == 2 + 10  # advancing time alone predicts nothing


def test_equal_contact_times_pick_the_lowest_rock_index() -> None:
    field = rocks((1060.0, 0.0, 0.0, 50.0), (1060.0, 0.0, 0.0, 50.0), cluster=(9, 2))
    w = make_world(field)
    spawn(w, vel=(100.0, 0.0, 0.0))
    w.run_until(w.horizon_us)
    assert w.events[0].data["rock"] == 0
