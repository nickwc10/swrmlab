"""Build a ready ``World`` from a scenario and a seed."""

import math

import numpy as np
from numpy.typing import NDArray

from swarmsim.core.engine import World
from swarmsim.core.geometry import Rocks
from swarmsim.core.rng import Stream, uniform
from swarmsim.core.segments import SegmentStore
from swarmsim.scenario.maps import build_map
from swarmsim.scenario.schema import Scenario, ShipClass, TeamSpec

# Tries per unit before giving up. Each try is one fixed-size draw.
ATTEMPTS = 100
# Draw indices in Stream.SPAWN: 0 and 1 pick the hiding rocks (see maps.build_map);
# placement draws start here, with a separate block per side.
_PLACE_BASE = 1000
# Room for 1 + 20 groups x 1000 ships slots of ATTEMPTS tries each.
_SIDE_STRIDE = 10_000_000
_OUTPOST_SLOT = 0


def _direction(u: NDArray[np.float64]) -> NDArray[np.float64]:
    """A unit vector, uniform on the sphere, from two uniform values."""
    cos_t = 2.0 * u[0] - 1.0
    sin_t = math.sqrt(max(0.0, 1.0 - cos_t * cos_t))
    phi = 2.0 * math.pi * u[1]
    return np.array([sin_t * math.cos(phi), sin_t * math.sin(phi), cos_t])


def _clear_gap(rocks: Rocks, pos: NDArray[np.float64], radius: float, skip: int = -1) -> float:
    """Smallest gap between a sphere at ``pos`` and any rock surface, ignoring ``skip``."""
    gap = np.linalg.norm(rocks.centers - pos, axis=1) - rocks.radii - radius
    if skip >= 0:
        gap[skip] = np.inf
    return float(gap.min())


def _draw(seed: int, side: int, slot: int, attempt: int) -> NDArray[np.float64]:
    index = _PLACE_BASE + side * _SIDE_STRIDE + slot * ATTEMPTS + attempt
    return uniform(seed, Stream.SPAWN, index, (4,))


def _spawn_team(
    world: World,
    team_name: str,
    team: TeamSpec,
    classes: dict[str, ShipClass],
    hide: int,
    side: int,
    seed: int,
    clearance: float,
    spread: float,
) -> None:
    rocks = world.rocks
    outpost = team.outpost
    pull = -1.0 if side == 1 else 1.0  # toward the enemy along x
    for attempt in range(ATTEMPTS):
        u = _draw(seed, side, _OUTPOST_SLOT, attempt)
        reach = rocks.radii[hide] + outpost.radius + clearance
        pos = rocks.centers[hide] + reach * _direction(u)
        if _clear_gap(rocks, pos, outpost.radius, skip=hide) > clearance:
            break
    else:
        raise ValueError(
            f"could not place the {team_name} outpost clear of other rocks after {ATTEMPTS} "
            "tries; reduce outpost_clearance or home_rocks, or increase cluster_radius"
        )
    outpost_pos = pos
    world.spawn(
        0, pos, (0.0, 0.0, 0.0), team=team_name, cls="outpost",
        dry_mass=outpost.dry_mass, fuel_mass=0.0, max_acc=0.0, ve=outpost.ve,
        radius=outpost.radius,
    )  # fmt: skip
    slot = _OUTPOST_SLOT
    for group in team.groups:
        cls = classes[group.ship_class]
        base = outpost.radius + cls.radius + clearance
        for _ in range(group.count):
            slot += 1
            for attempt in range(ATTEMPTS):
                u = _draw(seed, side, slot, attempt)
                pos = outpost_pos + (base + u[3] * spread) * _direction(u)
                if _clear_gap(rocks, pos, cls.radius) > clearance:
                    break
            else:
                raise ValueError(
                    f"could not place a {team_name} {group.ship_class} clear of rocks after "
                    f"{ATTEMPTS} tries; reduce ship_spread or outpost_clearance"
                )
            world.spawn(
                0, pos, (pull * group.initial_speed, 0.0, 0.0), team=team_name,
                cls=group.ship_class, dry_mass=cls.dry_mass, fuel_mass=cls.fuel_mass,
                max_acc=cls.max_acc, ve=cls.ve, radius=cls.radius,
            )  # fmt: skip


def build_world(scenario: Scenario, seed: int, swap: bool = False) -> World:
    """Make the map, then spawn blue's outpost and ships, then red's.

    Spawn order depends only on team, so unit ids do. Positions depend only on side (west
    0, east 1) and the seed, so with ``swap`` the two teams trade places exactly.
    """
    spec = scenario.map
    game_map = build_map(spec, seed, swap)
    world = World(
        SegmentStore(),
        game_map.rocks,
        scenario.c,
        horizon_us=round(scenario.duration * 1e6),
    )
    blue_side = 1 if swap else 0
    teams = (
        ("blue", scenario.blue, game_map.blue_hide, blue_side),
        ("red", scenario.red, game_map.red_hide, 1 - blue_side),
    )
    for name, team, hide, side in teams:
        _spawn_team(
            world, name, team, scenario.ship_classes, hide, side, seed,
            spec.outpost_clearance, spec.ship_spread,
        )  # fmt: skip
    return world
