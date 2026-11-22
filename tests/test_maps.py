import math

import numpy as np
import pytest

from swarmsim.core.rng import Stream, uniform
from swarmsim.scenario.maps import build_map
from swarmsim.scenario.schema import MapSpec

SMALL = MapSpec(
    separation=1e8,
    cluster_radius=1e6,
    rock_radius_min=1e3,
    rock_radius_max=2e4,
    home_rocks=20,
    hide_candidates=10,
    mid_clusters=2,
    mid_rocks=10,
)


def test_same_seed_same_map_and_different_seeds_differ() -> None:
    a, b, c = build_map(SMALL, 5), build_map(SMALL, 5), build_map(SMALL, 6)
    assert np.array_equal(a.rocks.centers, b.rocks.centers)
    assert np.array_equal(a.rocks.radii, b.rocks.radii)
    assert np.array_equal(a.rocks.cluster, b.rocks.cluster)
    assert (a.blue_hide, a.red_hide) == (b.blue_hide, b.red_hide)
    assert not np.array_equal(a.rocks.centers, c.rocks.centers)


def test_rock_counts_and_cluster_ids() -> None:
    m = build_map(SMALL, 1)
    n = 20 + 2 * 10
    assert m.rocks.centers.shape == (2 * n, 3)
    assert len(np.unique(m.rocks.cluster)) == 2 * (1 + 2)
    assert m.rocks.cluster[0] != m.rocks.cluster[n]  # west home vs east home
    assert np.all(m.rocks.cluster[:20] == m.rocks.cluster[0])
    assert np.all(m.rocks.cluster[n : n + 20] == m.rocks.cluster[n])
    # the home clusters are centered near x = -+separation / 2
    assert abs(m.rocks.centers[:20, 0].mean() + 5e7) < 1e6
    assert abs(m.rocks.centers[n : n + 20, 0].mean() - 5e7) < 1e6


def test_rocks_mirror_under_x() -> None:
    for seed in range(5):
        m = build_map(SMALL, seed)
        n = len(m.rocks.radii) // 2
        west, east = m.rocks.centers[:n], m.rocks.centers[n:]
        assert np.array_equal(east, west * np.array([-1.0, 1.0, 1.0]))
        assert np.array_equal(m.rocks.radii[:n], m.rocks.radii[n:])
        assert np.all(west[:, 0] + m.rocks.radii[:n] < 0.0)


def test_rocks_stay_in_requested_size_range_and_never_overlap() -> None:
    for seed in range(5):
        m = build_map(SMALL, seed)
        r = m.rocks.radii
        assert r.min() >= 1e3 and r.max() <= 2e4
        d = np.linalg.norm(m.rocks.centers[:, None] - m.rocks.centers[None], axis=2)
        gap = d - r[:, None] - r[None]
        np.fill_diagonal(gap, np.inf)
        assert gap.min() > 0.0


def test_mid_clusters_lie_between_home_and_middle() -> None:
    m = build_map(SMALL, 3)
    n = len(m.rocks.radii) // 2
    for cid in np.unique(m.rocks.cluster[:n])[1:]:
        x = m.rocks.centers[:n][m.rocks.cluster[:n] == cid, 0]
        assert -5e7 < x.mean() < 0.0


def test_hiding_rocks_are_home_candidates_and_overlap_nothing() -> None:
    for seed in range(20):
        m = build_map(SMALL, seed)
        n = len(m.rocks.radii) // 2
        assert 0 <= m.blue_hide < 10
        assert n <= m.red_hide < n + 10
        for h in (m.blue_hide, m.red_hide):
            d = np.linalg.norm(m.rocks.centers - m.rocks.centers[h], axis=1)
            gap = d - m.rocks.radii - m.rocks.radii[h]
            gap[h] = np.inf
            assert gap.min() > 0.0


def test_swap_exchanges_the_hiding_rocks_and_nothing_else() -> None:
    for seed in range(20):
        a, b = build_map(SMALL, seed), build_map(SMALL, seed, swap=True)
        assert (b.blue_hide, b.red_hide) == (a.red_hide, a.blue_hide)
        assert np.array_equal(a.rocks.centers, b.rocks.centers)
        assert np.array_equal(a.rocks.radii, b.rocks.radii)
        assert np.array_equal(a.rocks.cluster, b.rocks.cluster)


def _binomial_interval(n: int, p: float, tail: float) -> tuple[int, int]:
    pmf = [math.comb(n, k) * p**k * (1 - p) ** (n - k) for k in range(n + 1)]
    lo, acc = 0, 0.0
    while acc + pmf[lo] <= tail:
        acc += pmf[lo]
        lo += 1
    hi, acc = n, 0.0
    while acc + pmf[hi] <= tail:
        acc += pmf[hi]
        hi -= 1
    return lo, hi


def test_red_mirrors_blue_about_as_often_as_chance() -> None:
    n_seeds, candidates = 200, 10
    n = SMALL.home_rocks + SMALL.mid_clusters * SMALL.mid_rocks
    hits = 0
    for seed in range(n_seeds):
        m = build_map(SMALL, seed)
        hits += m.red_hide == m.blue_hide + n
    lo, hi = _binomial_interval(n_seeds, 1 / candidates, 0.0005)
    assert lo <= hits <= hi, f"{hits} mirrored of {n_seeds}; 99.9% interval is [{lo}, {hi}]"


def test_hiding_picks_are_independent_of_the_rest_of_the_seed_stream() -> None:
    # side 0 and side 1 use separate SPAWN draws; the pick is uniform over candidates
    for seed in range(10):
        m = build_map(SMALL, seed)
        n = len(m.rocks.radii) // 2
        west = int(uniform(seed, Stream.SPAWN, 0, (1,))[0] * 10)
        east = int(uniform(seed, Stream.SPAWN, 1, (1,))[0] * 10)
        assert (m.blue_hide, m.red_hide) == (west, n + east)


def test_unplaceable_rocks_raise_naming_the_setting() -> None:
    crowded = MapSpec(
        separation=1e7,
        cluster_radius=1e4,
        rock_radius_min=5e3,
        rock_radius_max=1e4,
        home_rocks=200,
        hide_candidates=10,
        mid_clusters=0,
    )
    with pytest.raises(ValueError, match="home_rocks"):
        build_map(crowded, 0)


def test_negative_seed_is_rejected() -> None:
    with pytest.raises(ValueError, match="seed"):
        build_map(SMALL, -1)
