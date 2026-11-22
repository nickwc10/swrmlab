import math

import numpy as np
import pytest

from swarmsim.core import geometry
from swarmsim.core.geometry import Rocks, first_within, occluded

Z = (0.0, 0.0, 0.0)


def test_long_range_closing_hits_at_expected_time() -> None:
    t = first_within((3e9, 0, 0), (-5e4, 0, 0), Z, 100.0, 1e6)
    assert t is not None
    assert abs(t - 59999.998) < 1e-3


def test_from_rest_with_closing_acceleration() -> None:
    t = first_within((3e8, 0, 0), Z, (-1.0, 0, 0), 1000.0, 1e6)
    assert t is not None
    assert abs(t - 24494.8566) < 1e-3


@pytest.mark.parametrize("miss", [1.0, 0.01])
def test_grazing_pass_that_misses_returns_none(miss: float) -> None:
    # Flies past the target at distance radius + miss.
    assert first_within((1e6, 100.0 + miss, 0), (-1e3, 0, 0), Z, 100.0, 1e4) is None


def test_grazing_pass_that_just_hits_returns_time() -> None:
    t = first_within((1e6, 100.0 - 0.01, 0), (-1e3, 0, 0), Z, 100.0, 1e4)
    assert t is not None
    assert 999.0 < t < 1001.0


def test_parked_returns_none() -> None:
    assert first_within((1e3, 0, 0), Z, Z, 10.0, 1e6) is None


def test_receding_returns_none() -> None:
    assert first_within((1e3, 0, 0), (10.0, 0, 0), Z, 10.0, 1e6) is None


def test_already_inside_returns_zero() -> None:
    assert first_within((5.0, 0, 0), (100.0, 0, 0), Z, 10.0, 1e6) == 0.0


def test_hit_later_than_dt_max_is_none_and_exactly_at_dt_max_counts() -> None:
    args = ((1000.0, 0, 0), (-10.0, 0, 0), Z, 100.0)  # hit at 90 s
    assert first_within(*args, 89.0) is None
    t = first_within(*args, 90.0)
    assert t is not None
    assert abs(t - 90.0) < 1e-3


def test_zero_radius_point_contact() -> None:
    t = first_within((100.0, 0, 0), (-10.0, 0, 0), Z, 0.0, 100.0)
    assert t is not None
    assert abs(t - 10.0) < 1e-3


def test_step_cap_returns_current_time_as_hit(monkeypatch: pytest.MonkeyPatch) -> None:
    args = ((1e6, 101.0, 0), (-1e3, 0, 0), Z, 100.0, 1e4)
    assert first_within(*args) is None
    monkeypatch.setattr(geometry, "_MAX_STEPS", 2)
    t = first_within(*args)
    assert t is not None
    assert 0.0 < t < 1e4


@pytest.mark.parametrize(
    "dr, dv, da, radius, dt_max",
    [
        ((1.0, 2.0), Z, Z, 1.0, 1.0),
        ((math.nan, 0, 0), Z, Z, 1.0, 1.0),
        (Z, (math.inf, 0, 0), Z, 1.0, 1.0),
        (Z, Z, (0, 0, math.nan), 1.0, 1.0),
        (Z, Z, Z, -1.0, 1.0),
        (Z, Z, Z, math.nan, 1.0),
        (Z, Z, Z, 1.0, -1.0),
        (Z, Z, Z, 1.0, math.inf),
    ],
)
def test_first_within_rejects_bad_inputs(
    dr: tuple[float, ...],
    dv: tuple[float, ...],
    da: tuple[float, ...],
    radius: float,
    dt_max: float,
) -> None:
    with pytest.raises(ValueError):
        first_within(dr, dv, da, radius, dt_max)  # type: ignore[arg-type]


def test_matches_dense_sampling_at_short_range() -> None:
    rng = np.random.default_rng(12345)
    times = np.linspace(0.0, 10.0, 100_001)
    hits = 0
    for _ in range(400):
        dr = rng.uniform(-60, 60, 3)
        dv = -dr * rng.uniform(0.02, 0.3) + rng.uniform(-8, 8, 3)  # aimed roughly at the target
        da = rng.uniform(-2, 2, 3)
        radius = float(rng.uniform(3, 25))
        t = first_within(tuple(dr), tuple(dv), tuple(da), radius, 10.0)  # type: ignore[arg-type]
        gap = np.linalg.norm(dr + np.outer(times, dv) + 0.5 * np.outer(times**2, da), axis=1)
        inside = np.flatnonzero(gap <= radius)
        if t is None:
            assert inside.size == 0
        else:
            hits += 1
            assert inside.size > 0
            assert times[inside[0]] >= t - 1e-6
            assert times[inside[0]] - t < 0.05  # not wildly early either
    assert 30 < hits < 370  # the sample exercises both outcomes


def test_rocks_validates_and_is_read_only() -> None:
    centers = np.zeros((2, 3))
    radii = np.array([1.0, 2.0])
    cluster = np.array([0, 1])
    rocks = Rocks(centers, radii, cluster)
    centers[0, 0] = 5.0
    assert rocks.centers[0, 0] == 0.0
    assert not rocks.centers.flags.writeable
    assert not rocks.radii.flags.writeable
    assert not rocks.cluster.flags.writeable
    assert rocks.centers.dtype == np.float64
    assert rocks.cluster.dtype == np.int64


def test_rocks_allows_empty() -> None:
    rocks = Rocks(np.zeros((0, 3)), np.zeros(0), np.zeros(0, dtype=np.int64))
    assert rocks.radii.shape == (0,)


@pytest.mark.parametrize(
    "centers, radii, cluster",
    [
        (np.zeros((2, 2)), np.ones(2), np.zeros(2)),
        (np.zeros((2, 3)), np.ones(3), np.zeros(2)),
        (np.zeros((2, 3)), np.ones(2), np.zeros(3)),
        (np.full((1, 3), np.nan), np.ones(1), np.zeros(1)),
        (np.zeros((1, 3)), np.array([np.inf]), np.zeros(1)),
        (np.zeros((1, 3)), np.array([0.0]), np.zeros(1)),
        (np.zeros((1, 3)), np.array([-1.0]), np.zeros(1)),
    ],
)
def test_rocks_rejects_bad_input(
    centers: np.ndarray, radii: np.ndarray, cluster: np.ndarray
) -> None:
    with pytest.raises(ValueError):
        Rocks(centers, radii, cluster)


def _rock(x: float = 0.0, y: float = 0.0, z: float = 0.0, r: float = 10.0):
    return np.array([[x, y, z]]), np.array([r])


def test_rock_between_blocks() -> None:
    c, r = _rock()
    assert occluded(np.array([-100.0, 0, 0]), np.array([100.0, 0, 0]), c, r)


def test_rock_beside_does_not_block() -> None:
    c, r = _rock(y=50.0)
    assert not occluded(np.array([-100.0, 0, 0]), np.array([100.0, 0, 0]), c, r)


def test_rock_behind_either_end_does_not_block() -> None:
    a = np.array([0.0, 0, 0])
    b = np.array([100.0, 0, 0])
    for x in (-50.0, 150.0):
        c, r = _rock(x=x)
        assert not occluded(a, b, c, r)


def test_tangent_counts_as_blocked() -> None:
    c, r = _rock(y=10.0)
    assert occluded(np.array([-100.0, 0, 0]), np.array([100.0, 0, 0]), c, r)


def test_endpoint_inside_or_on_rock_is_blocked() -> None:
    c, r = _rock()
    far = np.array([100.0, 0, 0])
    assert occluded(np.array([3.0, 0, 0]), far, c, r)
    assert occluded(np.array([10.0, 0, 0]), far, c, r)


def test_coincident_endpoints() -> None:
    c, r = _rock()
    assert occluded(np.array([5.0, 0, 0]), np.array([5.0, 0, 0]), c, r)
    assert not occluded(np.array([50.0, 0, 0]), np.array([50.0, 0, 0]), c, r)


def test_no_rocks_is_false() -> None:
    assert not occluded(np.zeros(3), np.ones(3), np.zeros((0, 3)), np.zeros(0))


def test_any_of_several_rocks_blocks() -> None:
    centers = np.array([[0.0, 500.0, 0.0], [50.0, 0.0, 0.0]])
    radii = np.array([10.0, 10.0])
    assert occluded(np.array([0.0, 0, 0]), np.array([100.0, 0, 0]), centers, radii)


def test_precision_at_3e9_m() -> None:
    a = np.array([3e9, 3e9, 3e9])
    b = np.array([3e9 + 1000.0, 3e9, 3e9])
    mid = 3e9 + 500.0
    assert occluded(a, b, np.array([[mid, 3e9 + 99.0, 3e9]]), np.array([100.0]))
    assert not occluded(a, b, np.array([[mid, 3e9 + 101.0, 3e9]]), np.array([100.0]))
