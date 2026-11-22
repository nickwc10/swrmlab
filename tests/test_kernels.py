"""The compiled kernels must give what the reference functions give."""

import math
from dataclasses import dataclass

import numpy as np
import pytest
from numpy.typing import NDArray

import swarmsim.core.geometry as geometry
from swarmsim.core.geometry import first_within
from swarmsim.core.kernels import (
    NO_END,
    NOT_SEEN,
    emission_time_batch,
    first_within_batch,
    pack,
)
from swarmsim.core.light import C, emission_time
from swarmsim.core.segments import SegmentStore

N = 10_000
_MAX_ACC = 100.0


# ---- emission_time_batch ----


def _random_store(rng: np.random.Generator, c: float, n_units: int) -> SegmentStore:
    """Units with 1 to 60 segments, |acc| up to 100 m/s2, always slower than 0.5 c.

    About 40% end. One that does not end coasts last, so it never outruns light.
    """
    store = SegmentStore()
    limit = 0.5 * c
    for i in range(n_units):
        t = int(rng.integers(-(10**9), 10**9))
        vel = rng.normal(size=3)
        vel *= rng.uniform(0.0, 0.3 * c) / np.linalg.norm(vel)
        unit = store.add_unit(t, rng.uniform(-1e9, 1e9, 3), vel, (0.0, 0.0, 0.0))
        n_seg = 1 if i % 4 == 0 else int(rng.integers(2, 61))
        for _ in range(n_seg - 1):
            _, vel = store.state(unit, t)
            acc = rng.normal(size=3)
            acc *= rng.uniform(0.0, _MAX_ACC) / np.linalg.norm(acc)
            # Seconds this burn can last before the speed limit.
            room = (limit - float(np.linalg.norm(vel))) / float(np.linalg.norm(acc))
            store.set_acc(unit, t, acc)
            t += 1 + int(1e6 * min(room * 0.5, 10.0 ** rng.uniform(-3.0, 4.0)))
        if rng.random() < 0.4:
            store.end(unit, t + int(rng.integers(0, 10**9)) if n_seg == 1 else t)
        elif n_seg > 1:
            store.set_acc(unit, t, (0.0, 0.0, 0.0))
    return store


@dataclass
class _Queries:
    unit: NDArray[np.int64]
    t_obs: NDArray[np.int64]
    pos: NDArray[np.float64]


def _random_queries(rng: np.random.Generator, store: SegmentStore, c: float, n: int) -> _Queries:
    """Observations from before birth to long after the end, many of them on a light-cone edge."""
    table = store.table()
    unit = rng.integers(0, len(store), n)
    t_obs, pos = np.zeros(n, dtype=np.int64), np.zeros((n, 3))
    for i in range(n):
        u = int(unit[i])
        t0s = table["t0"][table["unit"] == u]
        birth, end = store.birth(u), store.end_time(u)
        last = int(t0s[-1]) if end is None else end
        kind = rng.integers(0, 4)
        if kind == 0:  # a light-cone edge at birth, end or a segment boundary, +-2 us
            t_emit = int(rng.choice([birth, last, int(rng.choice(t0s))]))
        else:
            t_emit = int(rng.integers(birth, last + 1 + 10**9 if end is None else last + 1))
        there, _ = store.state(u, t_emit)
        away = rng.normal(size=3)
        away *= 10.0 ** rng.uniform(-1.0, 12.0) / np.linalg.norm(away)
        pos[i] = there + away
        t_obs[i] = t_emit + round(float(np.linalg.norm(away)) / c * 1e6)
        if kind == 0:
            t_obs[i] += int(rng.integers(-2, 3))
        elif kind == 1:  # anywhere, including before birth
            t_obs[i] = int(rng.integers(birth - 10**9, last + 10**10))
        elif kind == 2 and rng.random() < 0.2:  # observer exactly at the unit
            pos[i], t_obs[i] = there, t_emit
    return _Queries(unit, t_obs, pos)


def test_emission_times_equal_the_reference_on_10000_random_cases() -> None:
    rng = np.random.default_rng(20261122)
    speeds = [C, 1e6, 3e7, 2.5e8, 1e9]
    seen = {"before_birth": 0, "no_light_yet": 0, "end_seen": 0, "at_t_obs": 0}
    seen |= {"light": 0, "multi": 0, "ended_unit": 0}
    for c in speeds:
        store = _random_store(rng, c, 100)
        q = _random_queries(rng, store, c, N // len(speeds))
        paths = pack(store)
        got = emission_time_batch(paths, q.unit, q.t_obs, q.pos, c)
        assert got.dtype == np.int64
        for i in range(len(got)):
            u, t = int(q.unit[i]), int(q.t_obs[i])
            want = emission_time(store, u, t, q.pos[i], c)
            assert int(got[i]) == (NOT_SEEN if want is None else want), (c, i)
            end = store.end_time(u)
            if want is None:
                if t < store.birth(u):
                    kind = "before_birth"
                else:  # ended units can also be too far away for first light
                    there, _ = store.state(u, store.birth(u))
                    d = float(np.linalg.norm(there - q.pos[i]))
                    late = d > c * (t - store.birth(u)) / 1e6
                    kind = "no_light_yet" if late else "end_seen"
            else:
                kind = "at_t_obs" if want == t else "light"
                seen["multi"] += int(paths.starts[u + 1] - paths.starts[u] > 1)
                seen["ended_unit"] += int(end is not None)
            seen[kind] += 1
    # Every rule was exercised many times, not just the common path.
    assert min(seen.values()) >= 50, seen
    assert seen["multi"] > 1000 and seen["ended_unit"] > 1000, seen


def test_emission_time_of_a_still_unit_is_exact() -> None:
    store = SegmentStore()
    store.add_unit(0, (C, 0.0, 0.0), (0.0, 0.0, 0.0), (0.0, 0.0, 0.0))
    here = np.zeros((3, 3))
    got = emission_time_batch(pack(store), [0, 0, 0], [5_000_000, 1_000_000, 999_999], here)
    assert got.tolist() == [4_000_000, 0, NOT_SEEN]


def test_emission_time_rules_for_an_ended_unit() -> None:
    store = SegmentStore()
    store.add_unit(10, (C, 0.0, 0.0), (0.0, 0.0, 0.0), (0.0, 0.0, 0.0))
    store.end(0, 2_000_000)
    t_obs = [9, 1_000_009, 1_000_010, 2_999_999, 3_000_000]
    got = emission_time_batch(pack(store), [0] * 5, t_obs, np.zeros((5, 3)))
    assert got.tolist() == [NOT_SEEN, NOT_SEEN, 10, 1_999_999, NOT_SEEN]


def test_emission_time_for_an_observer_at_the_unit() -> None:
    store = SegmentStore()
    store.add_unit(10, (1.0, 2.0, 3.0), (0.0, 0.0, 0.0), (0.0, 0.0, 0.0))
    store.end(0, 50)
    t_obs = [9, 10, 49, 50, 51]
    got = emission_time_batch(pack(store), [0] * 5, t_obs, [[1.0, 2.0, 3.0]] * 5)
    assert got.tolist() == [NOT_SEEN, 10, 49, NOT_SEEN, NOT_SEEN]


def test_emission_time_at_the_extreme_supported_times() -> None:
    store = SegmentStore()
    store.add_unit(-(2**61), (0.0, 0.0, 0.0), (1e-3, 0.0, 0.0), (0.0, 0.0, 0.0))
    pos = np.array([[1e15, 0.0, 0.0]])
    got = emission_time_batch(pack(store), [0], [2**61], pos)
    assert got.tolist() == [emission_time(store, 0, 2**61, pos[0])]


def test_pack_lays_out_every_segment_birth_and_end() -> None:
    store = SegmentStore()
    store.add_unit(5, (1.0, 2.0, 3.0), (4.0, 5.0, 6.0), (0.0, 0.0, 0.0))
    store.add_unit(7, (0.0, 0.0, 0.0), (0.0, 0.0, 0.0), (0.0, 0.0, 0.0))
    store.set_acc(0, 20, (7.0, 8.0, 9.0))
    store.end(1, 30)
    paths = pack(store)
    assert paths.starts.tolist() == [0, 2, 3]
    assert paths.t0.tolist() == [5, 20, 7]
    assert paths.birth.tolist() == [5, 7]
    assert paths.end.tolist() == [NO_END, 30]
    assert paths.motion[0].tolist() == [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 0.0, 0.0, 0.0]
    assert paths.motion[1, 6:].tolist() == [7.0, 8.0, 9.0]


def test_empty_batches_are_allowed() -> None:
    none = np.zeros((0, 3))
    assert emission_time_batch(pack(SegmentStore()), [], [], none).shape == (0,)
    assert first_within_batch(none, none, none, [], []).shape == (0,)


@pytest.mark.parametrize(
    ("unit", "t_obs", "pos", "c", "match"),
    [
        ([1], [0], [[0.0, 0.0, 0.0]], C, "unit"),
        ([-1], [0], [[0.0, 0.0, 0.0]], C, "unit"),
        ([0.5], [0], [[0.0, 0.0, 0.0]], C, "integers"),
        ([0], [0.5], [[0.0, 0.0, 0.0]], C, "integers"),
        ([0], [2**61 + 1], [[0.0, 0.0, 0.0]], C, "range"),
        ([0], [0, 1], [[0.0, 0.0, 0.0]], C, "shape"),
        ([0], [0], [0.0, 0.0, 0.0], C, "shape"),
        ([0], [0], [[0.0, 0.0, math.nan]], C, "finite"),
        ([0], [0], [[0.0, 0.0, 0.0]], 0.0, "c must"),
        ([0], [0], [[0.0, 0.0, 0.0]], math.inf, "c must"),
    ],
)
def test_emission_time_batch_rejects_bad_input(
    unit: list[int], t_obs: list[int], pos: list[list[float]], c: float, match: str
) -> None:
    store = SegmentStore()
    store.add_unit(0, (0.0, 0.0, 0.0), (0.0, 0.0, 0.0), (0.0, 0.0, 0.0))
    with pytest.raises(ValueError, match=match):
        emission_time_batch(pack(store), unit, t_obs, pos, c)


# ---- first_within_batch ----


@dataclass
class _Contacts:
    dr: NDArray[np.float64]
    dv: NDArray[np.float64]
    da: NDArray[np.float64]
    radius: NDArray[np.float64]
    dt_max: NDArray[np.float64]


def _unit_vector(rng: np.random.Generator) -> NDArray[np.float64]:
    v = rng.normal(size=3)
    return np.asarray(v / np.linalg.norm(v))


def _random_contacts(rng: np.random.Generator, n: int) -> _Contacts:
    dr, dv, da = np.zeros((n, 3)), np.zeros((n, 3)), np.zeros((n, 3))
    radius, dt_max = np.zeros(n), np.zeros(n)
    for i in range(n):
        kind = i % 5
        rad = 10.0 ** rng.uniform(-1.0, 5.0)
        dist = rad + 10.0 ** rng.uniform(-3.0, 8.0)
        speed = 10.0 ** rng.uniform(0.0, 6.0)
        acc = rng.uniform(0.0, _MAX_ACC)
        toward, side = _unit_vector(rng), _unit_vector(rng)
        side -= toward * (side @ toward)
        side /= np.linalg.norm(side)
        dr[i] = -toward * dist
        dv[i], da[i] = toward * speed, _unit_vector(rng) * acc
        if kind == 0:  # unrelated directions
            dv[i] = _unit_vector(rng) * speed
        elif kind == 1:  # graze: passes the edge by -1 m to +1 m, no acceleration
            miss = float(rng.choice([-1.0, 1.0])) * 10.0 ** rng.uniform(-5.0, 0.0)
            dr[i] += side * (rad + miss)
            da[i] = 0.0
        elif kind == 2:  # aimed but braking: may stop short of contact by very little
            da[i] = -toward * acc
            stop = speed * speed / (2.0 * max(acc, 1e-3))
            dr[i] = -toward * (rad + stop + float(rng.normal()))
        elif kind == 3:  # slow relative motion, or none at all
            dv[i] *= float(rng.choice([0.0, 1e-6]))
            da[i] *= float(rng.choice([0.0, 1e-3]))
        elif rng.random() < 0.2:  # starts inside
            dr[i] *= rng.uniform(0.0, 1.0)
        radius[i] = 0.0 if rng.random() < 0.02 else rad
        dt_max[i] = 0.0 if rng.random() < 0.02 else 10.0 ** rng.uniform(0.0, 5.0)
        if rng.random() < 0.2:  # window ending right at the contact
            hit = first_within(dr[i], dv[i], da[i], radius[i], dt_max[i])
            if hit is not None:
                dt_max[i] = hit * float(rng.choice([1.0 - 1e-9, 1.0, 1.0 + 1e-9]))
    return _Contacts(dr, dv, da, radius, dt_max)


def _capped_contacts(rng: np.random.Generator, n: int) -> _Contacts:
    """Grazes where rounding stalls the stepping, so the 100,000-step cap decides."""
    out = _random_contacts(rng, n)
    for i in range(n):
        rad, far = 10.0 ** rng.uniform(5.0, 9.0), 10.0 ** rng.uniform(12.0, 14.0)
        speed = 10.0 ** rng.uniform(5.0, 7.4)
        out.dr[i] = (-far, rad + 10.0 ** rng.uniform(-2.9, -2.0), 0.0)
        out.dv[i], out.da[i] = (speed, 0.0, 0.0), 0.0
        out.radius[i], out.dt_max[i] = rad, 3.0 * far / speed
    return out


def compare_first_within(q: _Contacts) -> tuple[int, int, int]:
    """Assert agreement with the reference; return (hits, misses, bit-identical hit times)."""
    got = first_within_batch(q.dr, q.dv, q.da, q.radius, q.dt_max)
    hits = same = 0
    for i in range(len(got)):
        rad, limit = float(q.radius[i]), float(q.dt_max[i])
        want = first_within(q.dr[i], q.dv[i], q.da[i], rad, limit)
        assert math.isnan(got[i]) == (want is None), i
        if want is not None:
            hits += 1
            same += int(float(got[i]) == want)
            assert abs(float(got[i]) - want) <= 1e-12 * want, i
    return hits, len(got) - hits, same


def test_first_within_equals_the_reference_on_10000_random_cases() -> None:
    hits, misses, same = compare_first_within(_random_contacts(np.random.default_rng(7), N))
    print(f"first_within: {hits} hits, {misses} misses, {same} times bit-identical")
    assert hits > 2000 and misses > 2000


def test_first_within_step_cap_matches_the_reference(monkeypatch: pytest.MonkeyPatch) -> None:
    q = _capped_contacts(np.random.default_rng(3), 20)
    hits, _, same = compare_first_within(q)
    print(f"cap cases: {hits} hits, {same} times bit-identical")
    # Prove these cases reach the cap: a lower cap changes what the reference returns.
    monkeypatch.setattr(geometry, "_MAX_STEPS", 50_000)
    capped = 0
    for i in range(len(q.radius)):
        rad, limit = float(q.radius[i]), float(q.dt_max[i])
        t = first_within(q.dr[i], q.dv[i], q.da[i], rad, limit)
        if t is not None:
            p = q.dr[i] + q.dv[i] * t
            capped += int(math.sqrt(float(p @ p)) - rad > geometry._TOLERANCE_M)
    assert capped >= 5


def test_first_within_known_times_and_misses() -> None:
    dr = [[-10.0, 0.0, 0.0], [-10.0, 0.0, 0.0], [0.5, 0.0, 0.0], [-10.0, 5.0, 0.0]]
    dv = [[1.0, 0.0, 0.0]] * 4
    got = first_within_batch(dr, dv, np.zeros((4, 3)), [1.0] * 4, [100.0, 5.0, 1.0, 100.0])
    assert got[0] == pytest.approx(9.0, abs=1e-3)
    assert math.isnan(got[1])  # window closes first
    assert got[2] == 0.0  # starts inside
    assert math.isnan(got[3])  # passes by


@pytest.mark.parametrize(
    ("change", "match"),
    [
        ({"dr": np.zeros((2, 3))}, "shape"),
        ({"dv": np.zeros(3)}, "shape"),
        ({"da": np.full((1, 3), np.nan)}, "finite"),
        ({"radius": [-1.0]}, "radius"),
        ({"dt_max": [math.inf]}, "dt_max"),
        ({"radius": [1.0, 2.0]}, "shape"),
    ],
)
def test_first_within_batch_rejects_bad_input(
    change: dict[str, NDArray[np.float64]], match: str
) -> None:
    args = {
        "dr": np.ones((1, 3)),
        "dv": np.ones((1, 3)),
        "da": np.ones((1, 3)),
        "radius": [1.0],
        "dt_max": [1.0],
    }
    with pytest.raises(ValueError, match=match):
        first_within_batch(**(args | change))
