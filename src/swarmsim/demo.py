"""Scripted crossing: every ship flies to the enemy's home cluster and stops there.

No sensing, no weapons, no decisions after launch. Each ship accelerates, coasts, flips
and brakes to rest beside the center of the enemy home cluster, which is public map
knowledge. Planning reads the team's own ships, its own outpost and the rocks of its own
home cluster. It never reads where the enemy outpost hides.
"""

import math
from itertools import pairwise

import numpy as np
from numpy.typing import NDArray

from swarmsim.core.engine import World
from swarmsim.core.geometry import first_within
from swarmsim.core.propulsion import top_speed
from swarmsim.core.rng import Stream, uniform
from swarmsim.core.segments import Vec3, evaluate_segment
from swarmsim.scenario import Scenario, build_world

# A burn plan: (start time, acceleration) pairs in time order. Each holds until the next.
Plan = list[tuple[int, Vec3]]

# Share of a crossing's time spent coasting. Scouts burn longer, so they arrive sooner.
COAST = {"scout": 0.1, "attacker": 0.4}
_DEFAULT_COAST = 0.25
# Ships aim this many cluster radii beside the enemy home center, clear of its rocks.
_AIM_OFFSET = 1.5
_FUEL_SHARE = 0.95  # share of a ship's delta-v a plan may use
_DURATIONS = 10_000  # crossing times tried, evenly spaced up to the horizon
_SIDESTEP_ANGLES = 8
_AIM_DRAW = 0  # draw index in Stream.POLICY_BLUE and Stream.POLICY_RED


def aim_point(
    scenario: Scenario, seed: int, stream: Stream, slot: int, n_ships: int, enemy_x: float
) -> Vec3:
    """Where a team's ship number ``slot`` of ``n_ships`` stops.

    Slots are spread evenly on a circle around the enemy home center, in the plane facing
    the crossing. The circle's rotation is drawn from the team's own stream.
    """
    turn = float(uniform(seed, stream, _AIM_DRAW, (1,))[0])
    angle = 2.0 * math.pi * (turn + slot / n_ships)
    r = _AIM_OFFSET * scenario.map.cluster_radius
    return np.array([enemy_x, r * math.cos(angle), r * math.sin(angle)])


def plan_crossing(
    pos: Vec3, vel: Vec3, target: Vec3, start_us: int, horizon_us: int,
    max_acc: float, delta_v: float, coast: float,
) -> Plan | None:  # fmt: skip
    """Quickest tried burn, coast, burn that ends at rest at ``target`` by the horizon.

    The two burns last equally long and ``coast`` is the share of the time between them.
    For a given duration, the two accelerations follow from arriving at ``target`` with
    zero velocity. Returns None if no duration keeps within ``max_acc`` and ``delta_v``.
    """
    total = (horizon_us - start_us) * np.arange(1, _DURATIONS + 1, dtype=np.int64) // _DURATIONS
    burn_us = np.floor(total * ((1.0 - coast) / 2.0)).astype(np.int64)
    total, burn_us = total[burn_us > 0], burn_us[burn_us > 0]
    burn = (burn_us / 1e6)[:, None]
    cruise = ((total - 2 * burn_us) / 1e6)[:, None]
    # Velocity change of each burn.
    first = (target - pos - vel * (1.5 * burn + cruise)) / (burn + cruise)
    second = -vel - first
    speed1, speed2 = np.linalg.norm(first, axis=1), np.linalg.norm(second, axis=1)
    fits = np.maximum(speed1, speed2) <= max_acc * burn[:, 0]
    fits &= speed1 + speed2 <= delta_v
    if not fits.any():
        return None
    k = int(np.argmax(fits))
    t1 = start_us + int(burn_us[k])
    t3 = start_us + int(total[k])
    t2 = t3 - int(burn_us[k])
    rest = np.zeros(3)
    return [(start_us, first[k] / burn[k]), (t1, rest), (t2, second[k] / burn[k]), (t3, rest)]


def _clear(
    pos: Vec3, vel: Vec3, plan: Plan, horizon_us: int, centers: Vec3, reach: NDArray[np.float64]
) -> bool:
    """True if flying ``plan`` touches none of the given rocks; ``reach`` adds ship radius."""
    ends = [*plan, (horizon_us, np.zeros(3))]
    for (t, acc), (t_next, _) in pairwise(ends):
        dt = (t_next - t) / 1e6
        for center, r in zip(centers, reach, strict=True):
            dr = (pos - center).tolist()
            if first_within(dr, vel.tolist(), acc.tolist(), float(r), dt) is not None:
                return False
        pos, vel = evaluate_segment(t, pos, vel, acc, t_next)
    return True


def _sidesteps(
    pos: Vec3, target: Vec3, center: Vec3, rock_radius: float
) -> list[tuple[Vec3, float]]:
    """Directions square to the route, the one away from the rock first, and how far to go."""
    along = (target - pos) / np.linalg.norm(target - pos)
    away = pos - center
    away = away - along * (away @ along)
    if np.linalg.norm(away) == 0.0:
        away = np.cross(along, (0.0, 0.0, 1.0) if abs(along[2]) < 0.9 else (1.0, 0.0, 0.0))
    u = away / np.linalg.norm(away)
    w = np.cross(along, u)
    angles = [2.0 * math.pi * k / _SIDESTEP_ANGLES for k in range(_SIDESTEP_ANGLES)]
    return [(u * math.cos(a) + w * math.sin(a), 1.5 * rock_radius) for a in angles]


def _plan_ship(world: World, scenario: Scenario, unit: int, outpost: int, target: Vec3) -> Plan:
    """A ship's burns from time 0. Empty if it cannot make the crossing: it then coasts."""
    cls_name = world.units[unit][2]
    cls = scenario.ship_classes[cls_name]
    pos, vel = world.store.state(unit, 0)
    coast = COAST.get(cls_name, _DEFAULT_COAST)
    delta_v = _FUEL_SHARE * top_speed(0.0, cls.dry_mass, cls.fuel_mass, cls.ve)
    horizon = world.horizon_us
    direct = plan_crossing(pos, vel, target, 0, horizon, cls.max_acc, delta_v, coast)
    if direct is None:
        return []
    # A ship may start behind its own rock: then step aside first.
    rocks = world.rocks
    home_pos, _ = world.store.state(outpost, 0)
    nearest = int(np.argmin(np.linalg.norm(rocks.centers - home_pos, axis=1) - rocks.radii))
    home = rocks.cluster == rocks.cluster[nearest]
    centers, reach = rocks.centers[home], rocks.radii[home] + cls.radius
    if _clear(pos, vel, direct, horizon, centers, reach):
        return direct
    options = _sidesteps(pos, target, rocks.centers[nearest], float(rocks.radii[nearest]))
    for side, distance in options:
        step_s = math.sqrt(2.0 * distance / cls.max_acc)
        step_us = math.ceil(step_s * 1e6)
        if step_us >= horizon:
            break
        acc = side * cls.max_acc
        p, v = evaluate_segment(0, pos, vel, acc, step_us)
        left = delta_v - cls.max_acc * step_us / 1e6
        rest = plan_crossing(p, v, target, step_us, horizon, cls.max_acc, left, coast)
        if rest is None:
            continue
        plan = [(0, acc), *rest]
        if _clear(pos, vel, plan, horizon, centers, reach):
            return plan
    return direct


def run_demo(scenario: Scenario, seed: int) -> World:
    """Build the world, send every ship across, and run to the scenario horizon.

    Blue starts in the west. Rocks between the home clusters are not avoided, so a ship
    can in principle hit one; that is logged like any collision.
    """
    world = build_world(scenario, seed)
    half = scenario.map.separation / 2.0
    for team, stream, enemy_x in (
        ("blue", Stream.POLICY_BLUE, half),
        ("red", Stream.POLICY_RED, -half),
    ):
        outpost, *ships = [u[0] for u in world.units if u[1] == team]
        for slot, unit in enumerate(ships):
            target = aim_point(scenario, seed, stream, slot, len(ships), enemy_x)
            for t, acc in _plan_ship(world, scenario, unit, outpost, target):
                world.command_acc(unit, t, acc)
    world.run_until(world.horizon_us)
    return world
