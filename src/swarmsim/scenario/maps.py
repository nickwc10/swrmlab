"""Seeded, mirrored asteroid maps.

Rocks are placed in the west half (x < 0) and mirrored to the east, so the two sides
face the same terrain. Rock order is: all west rocks (home cluster first, then the middle
clusters), then their mirrors in the same order. West cluster ``k`` has id ``k`` and its
mirror has id ``k + 1 + mid_clusters``.

All randomness comes from keyed draws (see ``core.rng``) of fixed size, so a change to
one setting never reshuffles an unrelated draw.
"""

import math
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from swarmsim.core.geometry import Rocks
from swarmsim.core.rng import Stream, uniform
from swarmsim.scenario.schema import MapSpec

# Tries per rock before giving up. Each try is one fixed-size draw.
ATTEMPTS = 100
# Draw indices: cluster centers use 0..999, rock tries start here.
_ROCK_BASE = 1000
_MAX_ROCKS_PER_CLUSTER = 5001  # stride between clusters; MapSpec caps rocks at 5000
_MIRROR = np.array([-1.0, 1.0, 1.0])


@dataclass(frozen=True)
class GameMap:
    """Rocks and the indices (into ``rocks``) of each team's hiding rock."""

    rocks: Rocks
    blue_hide: int
    red_hide: int


def _cluster_center(spec: MapSpec, seed: int, cluster: int) -> NDArray[np.float64]:
    if cluster == 0:
        return np.array([-spec.separation / 2.0, 0.0, 0.0])
    u = uniform(seed, Stream.MAP, cluster, (3,))
    lo = -spec.separation / 2.0 + 2.0 * spec.cluster_radius
    hi = -spec.cluster_radius - spec.rock_radius_max
    lateral = spec.mid_lateral
    return np.array([lo + u[0] * (hi - lo), (2 * u[1] - 1) * lateral, (2 * u[2] - 1) * lateral])


def _west_rocks(spec: MapSpec, seed: int) -> Rocks:
    counts = [spec.home_rocks] + [spec.mid_rocks] * spec.mid_clusters
    total = sum(counts)
    centers = np.zeros((total, 3))
    radii = np.zeros(total)
    cluster = np.zeros(total, dtype=np.int64)
    n = 0
    for k, count in enumerate(counts):
        middle = _cluster_center(spec, seed, k)
        setting = "home_rocks" if k == 0 else "mid_rocks"
        for slot in range(count):
            for attempt in range(ATTEMPTS):
                index = _ROCK_BASE + (k * _MAX_ROCKS_PER_CLUSTER + slot) * ATTEMPTS + attempt
                u = uniform(seed, Stream.MAP, index, (4,))
                reach = spec.cluster_radius * u[0] ** (1.0 / 3.0)
                cos_t = 2.0 * u[1] - 1.0
                sin_t = math.sqrt(max(0.0, 1.0 - cos_t * cos_t))
                phi = 2.0 * math.pi * u[2]
                c = middle + reach * np.array([sin_t * math.cos(phi), sin_t * math.sin(phi), cos_t])
                r = spec.rock_radius_min + u[3] * (spec.rock_radius_max - spec.rock_radius_min)
                if c[0] + r >= 0.0:
                    continue  # must lie wholly in the west half, so it cannot meet its mirror
                have_c, have_r = centers[:n], radii[:n]
                near = np.linalg.norm(have_c - c, axis=1) - have_r - r
                near_mirror = np.linalg.norm(have_c * _MIRROR - c, axis=1) - have_r - r
                if n and (near.min() <= 0.0 or near_mirror.min() <= 0.0):
                    continue
                centers[n], radii[n], cluster[n] = c, r, k
                n += 1
                break
            else:
                raise ValueError(
                    f"could not place rock {slot} of cluster {k} after {ATTEMPTS} tries; "
                    f"reduce {setting} or rock_radius_max, or increase cluster_radius"
                )
    return Rocks(centers, radii, cluster)


def build_map(spec: MapSpec, seed: int, swap: bool = False) -> GameMap:
    """Make the rocks and pick each team's hiding rock.

    The west and east hiding rocks are independent uniform picks among the first
    ``hide_candidates`` rocks of their home clusters, keyed by side (west 0, east 1). Blue
    is west unless ``swap``, which only exchanges the two picks.
    """
    west = _west_rocks(spec, seed)
    n = len(west.radii)
    n_clusters = 1 + spec.mid_clusters
    rocks = Rocks(
        np.concatenate([west.centers, west.centers * _MIRROR]),
        np.concatenate([west.radii, west.radii]),
        np.concatenate([west.cluster, west.cluster + n_clusters]),
    )
    k = spec.hide_candidates
    picks = []
    for side in (0, 1):
        u = float(uniform(seed, Stream.SPAWN, side, (1,))[0])
        picks.append(min(int(u * k), k - 1) + side * n)
    west_pick, east_pick = picks
    if swap:
        return GameMap(rocks, east_pick, west_pick)
    return GameMap(rocks, west_pick, east_pick)
