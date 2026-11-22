import math

import numpy as np
import pytest

from swarmsim.core.light import C, arrival_time, emission_time
from swarmsim.core.segments import SegmentStore


def _dist(store: SegmentStore, unit: int, t: int, p: tuple[float, float, float]) -> float:
    pos, _ = store.state(unit, t)
    dx, dy, dz = float(pos[0]) - p[0], float(pos[1]) - p[1], float(pos[2]) - p[2]
    return math.sqrt(dx * dx + dy * dy + dz * dz)


def _rest_unit(x: float = C, born: int = 0) -> tuple[SegmentStore, int]:
    s = SegmentStore()
    return s, s.add_unit(born, (x, 0, 0), (0, 0, 0), (0, 0, 0))


def test_rest_unit_one_light_second_away() -> None:
    s, u = _rest_unit(born=-5_000_000)
    t = emission_time(s, u, 1_000_000, (0, 0, 0))
    assert t is not None
    assert -1 <= t <= 0  # seen as of 1_000_000 us ago, within 1 us, never early


def test_emission_tight_and_never_early() -> None:
    s, u = _rest_unit(born=-5_000_000)
    t = emission_time(s, u, 1_000_000, (0, 0, 0))
    assert t is not None
    assert _dist(s, u, t, (0, 0, 0)) <= C * ((1_000_000 - t) / 1e6)
    assert _dist(s, u, t + 1, (0, 0, 0)) > C * ((1_000_000 - t - 1) / 1e6)


def test_before_first_light_and_before_birth() -> None:
    s, u = _rest_unit(born=0)
    # Born at t=0, 1 light-second away: nothing visible until t=1_000_000.
    assert emission_time(s, u, 999_999, (0, 0, 0)) is None
    assert emission_time(s, u, 1_000_000, (0, 0, 0)) == 0
    assert emission_time(s, u, -1, (0, 0, 0)) is None


def test_observer_at_unit_sees_now() -> None:
    s, u = _rest_unit(x=0.0)
    assert emission_time(s, u, 123_456, (0, 0, 0)) == 123_456


def test_after_end_seen_until_end_light_arrives() -> None:
    s, u = _rest_unit(born=-5_000_000)
    s.end(u, 100)
    # End's light reaches the observer at 100 + 1_000_000.
    t = emission_time(s, u, 500_000, (0, 0, 0))
    assert t is not None and t <= 100
    last = emission_time(s, u, 1_000_099, (0, 0, 0))
    assert last is not None and 99 <= last < 100
    assert emission_time(s, u, 1_000_100, (0, 0, 0)) is None
    assert emission_time(s, u, 5_000_000, (0, 0, 0)) is None


def test_adjustable_c_with_multi_segment_unit() -> None:
    c = 3e5
    s = SegmentStore()
    u = s.add_unit(0, (1000.0, 0, 0), (50.0, 0, 0), (0, 0, 0))
    s.set_acc(u, 2_000_000, (0, 20.0, 0))
    s.set_acc(u, 5_000_000, (-10.0, 0, 5.0))
    s.set_acc(u, 9_000_000, (0, 0, 0))
    obs = (-2000.0, 300.0, 50.0)
    for t_obs in (100_000, 3_000_000, 6_000_000, 12_000_000, 20_000_000):
        t = emission_time(s, u, t_obs, obs, c=c)
        assert t is not None
        assert _dist(s, u, t, obs) <= c * ((t_obs - t) / 1e6)
        if t < t_obs:
            assert _dist(s, u, t + 1, obs) > c * ((t_obs - t - 1) / 1e6)


def test_random_moving_units_never_early() -> None:
    rng = np.random.default_rng(12345)
    s = SegmentStore()
    c = 3e5
    units = []
    for _ in range(500):
        pos = rng.uniform(-1e5, 1e5, 3)
        vel = rng.uniform(-100, 100, 3)
        u = s.add_unit(0, pos, vel, rng.uniform(-5, 5, 3))
        for k in range(1, int(rng.integers(1, 4))):
            s.set_acc(u, k * 3_000_000, rng.uniform(-5, 5, 3))
        units.append(u)
    for u in units:
        obs = tuple(float(v) for v in rng.uniform(-1e5, 1e5, 3))
        t_obs = int(rng.integers(5_000_000, 60_000_000))
        t = emission_time(s, u, t_obs, obs, c=c)
        assert t is not None
        d = _dist(s, u, t, obs)
        assert d <= c * ((t_obs - t) / 1e6)
        if t < t_obs:
            assert _dist(s, u, t + 1, obs) > c * ((t_obs - t - 1) / 1e6)


def test_arrival_at_one_light_second() -> None:
    s, u = _rest_unit(born=-5_000_000)
    assert arrival_time(s, u, 0, (0, 0, 0), 10_000_000) == 1_000_000
    s2, u2 = _rest_unit(born=-5_000_000)
    assert arrival_time(s2, u2, 0, (0, 0, 0), 999_999) is None


def test_arrival_tight_moving_receiver_multi_segment() -> None:
    c = 3e5
    s = SegmentStore()
    u = s.add_unit(0, (3e5, 0, 0), (100.0, 0, 0), (0, 0, 0))
    s.set_acc(u, 500_000, (0, 30.0, 0))
    s.set_acc(u, 4_000_000, (-5.0, 0, 0))
    p = (10.0, -20.0, 5.0)
    t_send = 100_000
    t = arrival_time(s, u, t_send, p, 100_000_000, c=c)
    assert t is not None and t >= t_send
    assert _dist(s, u, t, p) <= c * ((t - t_send) / 1e6)
    assert _dist(s, u, t - 1, p) > c * ((t - 1 - t_send) / 1e6)


def test_arrival_receiver_at_send_position_is_instant() -> None:
    s, u = _rest_unit(x=5.0)
    assert arrival_time(s, u, 777, (5.0, 0, 0), 10_000) == 777


def test_arrival_respects_birth_end_and_horizon() -> None:
    s, u = _rest_unit(born=2_000_000)
    # Sent before birth: light already past; receiver gets it at birth.
    assert arrival_time(s, u, 0, (0, 0, 0), 10_000_000) == 2_000_000
    s2, u2 = _rest_unit(born=0)
    s2.end(u2, 500_000)
    assert arrival_time(s2, u2, 0, (0, 0, 0), 10_000_000) is None
    s3, u3 = _rest_unit(born=0)
    assert arrival_time(s3, u3, 0, (0, 0, 0), 999_999) is None
    assert arrival_time(s3, u3, 0, (0, 0, 0), 5_000_000) == 1_000_000


def test_arrival_random_tight() -> None:
    rng = np.random.default_rng(777)
    c = 3e5
    for _ in range(200):
        s = SegmentStore()
        u = s.add_unit(0, rng.uniform(-1e5, 1e5, 3), rng.uniform(-100, 100, 3), (0, 0, 0))
        s.set_acc(u, 2_000_000, rng.uniform(-5, 5, 3))
        p = tuple(float(v) for v in rng.uniform(-1e5, 1e5, 3))
        t_send = int(rng.integers(0, 5_000_000))
        t = arrival_time(s, u, t_send, p, 10**9, c=c)
        assert t is not None
        assert _dist(s, u, t, p) <= c * ((t - t_send) / 1e6)
        if t > t_send:
            assert _dist(s, u, t - 1, p) > c * ((t - 1 - t_send) / 1e6)


def test_validation() -> None:
    s, u = _rest_unit()
    for bad_c in (0.0, -1.0, math.inf, math.nan):
        with pytest.raises(ValueError):
            emission_time(s, u, 0, (0, 0, 0), c=bad_c)
        with pytest.raises(ValueError):
            arrival_time(s, u, 0, (0, 0, 0), 10, c=bad_c)
    for bad_p in ((0, 0), (0, 0, math.nan), (0, math.inf, 0)):
        with pytest.raises(ValueError):
            emission_time(s, u, 0, bad_p)
        with pytest.raises(ValueError):
            arrival_time(s, u, 0, bad_p, 10)
