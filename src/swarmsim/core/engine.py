"""Event engine: advances integer-microsecond time, burns fuel and ends units on rocks.

Events wait in a heap keyed ``(t_us, priority, sequence)``. At one microsecond the order is
collision, fuel_out, command. Each unit has a generation number that goes up whenever its
segment changes, so predictions made for an older segment are dropped when they come up.

Predictions always run to the horizon (the battle's end), never to the ``run_until``
target, so how a run is sliced cannot change its outcome.
"""

import math
import operator
from bisect import bisect_right
from collections.abc import Sequence
from dataclasses import dataclass, field
from heapq import heappop, heappush
from typing import Any, NamedTuple

import numpy as np
from numpy.typing import NDArray

from swarmsim.core.geometry import Rocks, first_within
from swarmsim.core.light import C, _check_c
from swarmsim.core.propulsion import burn_time, fuel_after
from swarmsim.core.segments import SegmentStore, _time, _vec3

# Priorities within one microsecond.
_COLLISION, _FUEL_OUT, _COMMAND = 0, 1, 2
_ACC_SLACK = 1e-12
# Largest acceleration a unit may be given [m/s2]; beyond this the float math loses meaning.
MAX_ACC = 1e9
_MAX_SPEED_FRACTION = 0.1

_Triple = tuple[float, float, float]
_ZERO: _Triple = (0.0, 0.0, 0.0)


@dataclass(frozen=True)
class Event:
    """Something that happened. ``other`` is a second unit's id where one is involved."""

    t_us: int
    type: str
    unit: int
    other: int | None
    data: dict[str, Any]


class _Queued(NamedTuple):
    """Heap entry. ``seq`` is unique, so comparison never reaches the later fields."""

    t_us: int
    priority: int
    seq: int
    unit: int
    gen: int  # segment generation a prediction was made for; unused by commands
    rock: int  # rock a collision was predicted with; -1 otherwise


@dataclass
class _Unit:
    team: str
    cls: str
    dry_mass: float
    max_acc: float
    ve: float
    radius: float
    empty: bool
    gen: int = 0
    # Mass history: start time, mass then, acceleration magnitude. Where several entries
    # share a start time, the last one counts.
    t0: list[int] = field(default_factory=list)
    m0: list[float] = field(default_factory=list)
    acc: list[float] = field(default_factory=list)


class _Cluster(NamedTuple):
    """Bounding sphere of one rock cluster and the indices of its rocks."""

    center: _Triple
    radius: float
    rocks: list[int]


def _triple(v: NDArray[np.float64]) -> _Triple:
    return float(v[0]), float(v[1]), float(v[2])


def _minus(a: _Triple, b: _Triple) -> _Triple:
    return a[0] - b[0], a[1] - b[1], a[2] - b[2]


def _bounding_spheres(rocks: Rocks) -> list[_Cluster]:
    clusters = []
    for cluster_id in np.unique(rocks.cluster):
        members = np.flatnonzero(rocks.cluster == cluster_id)
        centers = rocks.centers[members]
        middle = centers.mean(axis=0)
        reach = np.linalg.norm(centers - middle, axis=1) + rocks.radii[members]
        clusters.append(_Cluster(_triple(middle), float(reach.max()), members.tolist()))
    return clusters


class World:
    """Units flying through rocks from time 0 to ``horizon_us``.

    ``store`` must be empty; the world fills it and owns it from then on.
    """

    def __init__(self, store: SegmentStore, rocks: Rocks, c: float = C, *, horizon_us: int) -> None:
        if len(store) != 0:
            raise ValueError("store must be empty")
        self.store = store
        self.rocks = rocks
        self.c = _check_c(c)
        self.horizon_us = _time("horizon_us", horizon_us)
        if self.horizon_us < 0:
            raise ValueError(f"horizon_us must be >= 0, got {horizon_us}")
        self.events: list[Event] = []
        self._now = 0
        self._units: list[_Unit] = []
        self._heap: list[_Queued] = []
        self._seq = 0
        # Queued commands by (unit, time): the latest one's sequence number and acceleration.
        self._commands: dict[tuple[int, int], tuple[int, _Triple]] = {}
        self._spheres = [
            (_triple(p), float(r)) for p, r in zip(rocks.centers, rocks.radii, strict=True)
        ]
        self._clusters = _bounding_spheres(rocks)

    @property
    def now_us(self) -> int:
        return self._now

    @property
    def units(self) -> list[tuple[int, str, str, int, int | None]]:
        """(id, team, cls, birth, end) per unit in id order; end is None while it flies."""
        return [
            (i, u.team, u.cls, self.store.birth(i), self.store.end_time(i))
            for i, u in enumerate(self._units)
        ]

    def spawn(
        self,
        t_us: int,
        pos: Sequence[float] | NDArray[np.float64],
        vel: Sequence[float] | NDArray[np.float64],
        *,
        team: str,
        cls: str,
        dry_mass: float,
        fuel_mass: float,
        max_acc: float,
        ve: float,
        radius: float,
    ) -> int:
        """Add a coasting unit and return its id.

        A unit with ``t_us`` later than now is recorded at once, with that birth time: it
        gets its id and appears in ``units`` immediately, but has no state, mass or light
        before its birth and cannot be commanded for an earlier time.
        """
        t = self._check_future("t_us", t_us)
        p, v = _vec3("pos", pos), _vec3("vel", vel)
        dry, fuel, limit = float(dry_mass), float(fuel_mass), float(max_acc)
        ve, radius = float(ve), float(radius)
        for name, value in (("dry_mass", dry), ("ve", ve)):
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive, got {value}")
        for name, value in (("fuel_mass", fuel), ("max_acc", limit), ("radius", radius)):
            if not math.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be finite and >= 0, got {value}")
        if limit > MAX_ACC:
            raise ValueError(f"max_acc {limit} exceeds the limit {MAX_ACC} m/s2")
        # Every unit must stay slower than light for the light-delay search to hold.
        top_speed = math.hypot(*_triple(v)) + ve * math.log((dry + fuel) / dry)
        if top_speed > _MAX_SPEED_FRACTION * self.c:
            raise ValueError(f"top speed {top_speed} m/s exceeds {_MAX_SPEED_FRACTION} c")
        contact = self._first_contact(t, _triple(p), _triple(v), _ZERO, radius)
        if contact is not None and contact[0] == 0.0:
            raise ValueError(f"unit would start touching rock {contact[1]}")

        unit = self.store.add_unit(t, p, v, _ZERO)
        self._units.append(_Unit(team, cls, dry, limit, ve, radius, empty=fuel == 0.0))
        self._begin_mass_segment(unit, t, dry + fuel, 0.0)
        self._queue_collision(unit, t, contact)
        return unit

    def command_acc(self, unit: int, t_us: int, acc: Sequence[float] | NDArray[np.float64]) -> None:
        """Queue a burn at ``acc`` from ``t_us``; zero means coast.

        If several commands for one unit and instant are still queued, only the last
        applies. A command whose unit ends first is dropped.
        """
        unit = int(operator.index(unit))  # ids land in events, which must be JSON-safe
        birth, end = self.store.birth(unit), self.store.end_time(unit)
        t = self._check_future("t_us", t_us)
        if t < birth:
            raise ValueError(f"t_us {t} is before unit {unit} is born at {birth}")
        if end is not None:
            raise ValueError(f"unit {unit} already ended at {end}")
        a = _triple(_vec3("acc", acc))
        magnitude = math.hypot(*a)
        limit = self._units[unit].max_acc
        if magnitude > limit * (1.0 + _ACC_SLACK):
            raise ValueError(f"acceleration {magnitude} exceeds unit {unit}'s limit {limit}")
        seq = self._push(t, _COMMAND, unit)
        self._commands[(unit, t)] = (seq, a)

    def run_until(self, t_us: int) -> None:
        """Process every event up to and including ``t_us``, then stop the clock there."""
        t = self._check_future("t_us", t_us)
        while self._heap and self._heap[0].t_us <= t:
            entry = heappop(self._heap)
            self._now = entry.t_us
            self._process(entry)
        self._now = t

    def mass(self, unit: int, t_us: int) -> float:
        """Total mass at a time the unit has lived through (birth to min(now, end))."""
        birth, end = self.store.birth(unit), self.store.end_time(unit)
        t = _time("t_us", t_us)
        last = self._now if end is None else end
        if not birth <= t <= last:
            raise ValueError(f"t_us {t} is outside unit {unit}'s past [{birth}, {last}]")
        return self._mass(unit, t)

    def _check_future(self, name: str, t_us: int) -> int:
        t = _time(name, t_us)
        if not self._now <= t <= self.horizon_us:
            raise ValueError(f"{name} {t} is outside [{self._now}, {self.horizon_us}]")
        return t

    def _push(self, t: int, priority: int, unit: int, rock: int = -1) -> int:
        seq = self._seq
        self._seq += 1
        heappush(self._heap, _Queued(t, priority, seq, unit, self._units[unit].gen, rock))
        return seq

    def _log(self, t: int, type_: str, unit: int, data: dict[str, Any]) -> None:
        self.events.append(Event(t, type_, unit, None, data))

    def _process(self, entry: _Queued) -> None:
        if entry.priority == _COMMAND:
            self._apply_command(entry)
        elif entry.gen != self._units[entry.unit].gen:
            return  # predicted for a segment that has since been replaced
        elif entry.priority == _COLLISION:
            self._collide(entry)
        else:
            self._run_dry(entry)

    def _collide(self, entry: _Queued) -> None:
        self.store.end(entry.unit, entry.t_us)
        self._units[entry.unit].gen += 1
        self._log(entry.t_us, "collision", entry.unit, {"rock": entry.rock})
        self._log(entry.t_us, "end", entry.unit, {})

    def _run_dry(self, entry: _Queued) -> None:
        u = self._units[entry.unit]
        u.empty = True
        self._log(entry.t_us, "fuel_out", entry.unit, {})
        self._change_segment(entry.unit, entry.t_us, u.dry_mass, _ZERO)

    def _apply_command(self, entry: _Queued) -> None:
        seq, acc = self._commands.get((entry.unit, entry.t_us), (None, _ZERO))
        if seq != entry.seq:
            return  # a later command for the same unit and instant replaced this one
        del self._commands[(entry.unit, entry.t_us)]
        if self.store.end_time(entry.unit) is not None:
            return
        if self._units[entry.unit].empty:
            acc = _ZERO
        self._log(entry.t_us, "burn", entry.unit, {"acc": list(acc)})
        mass = self._mass(entry.unit, entry.t_us)
        self._change_segment(entry.unit, entry.t_us, mass, acc)

    def _change_segment(self, unit: int, t: int, mass: float, acc: _Triple) -> None:
        """Start a new segment, dropping old predictions and making new ones."""
        u = self._units[unit]
        self.store.set_acc(unit, t, acc)
        u.gen += 1
        magnitude = math.hypot(*acc)
        self._begin_mass_segment(unit, t, mass, magnitude)
        pos, vel = self.store.state(unit, t)
        contact = self._first_contact(t, _triple(pos), _triple(vel), acc, u.radius)
        self._queue_collision(unit, t, contact)
        burn_us = burn_time(mass, u.dry_mass, magnitude, u.ve) * 1e6
        if burn_us < self.horizon_us - t + 1:  # floor(burn_us) is within the horizon
            self._push(t + math.floor(burn_us), _FUEL_OUT, unit)

    def _begin_mass_segment(self, unit: int, t: int, mass: float, acc: float) -> None:
        u = self._units[unit]
        u.t0.append(t)
        u.m0.append(mass)
        u.acc.append(acc)

    def _mass(self, unit: int, t: int) -> float:
        u = self._units[unit]
        i = bisect_right(u.t0, t) - 1
        left = fuel_after(u.m0[i], u.acc[i], (t - u.t0[i]) / 1e6, u.ve)
        return max(u.dry_mass, left)  # rounding must not eat into the dry mass

    def _queue_collision(self, unit: int, t: int, contact: tuple[float, int] | None) -> None:
        if contact is None:
            return
        dt, rock = contact
        # Contact time rounds up; rounding must not carry it past the horizon.
        t_hit = min(t + math.ceil(dt * 1e6), self.horizon_us)
        self._push(t_hit, _COLLISION, unit, rock)

    def _first_contact(
        self, t: int, pos: _Triple, vel: _Triple, acc: _Triple, radius: float
    ) -> tuple[float, int] | None:
        """Earliest (seconds after t, rock index) contact up to the horizon.

        Clusters whose bounding sphere is never reached are skipped. The sphere is padded
        so that rounding and the tolerance in ``first_within`` can never skip a rock that a
        direct check would have hit. Rocks are always checked over the whole window, so a
        result does not depend on how rocks are grouped or on the order of checks.
        """
        dt_max = (self.horizon_us - t) / 1e6
        # Size of the terms summed inside first_within, for the rounding part of the pad.
        travel = math.hypot(*vel) * dt_max + 0.5 * math.hypot(*acc) * dt_max * dt_max
        best: tuple[float, int] | None = None
        for cluster in self._clusters:
            dr = _minus(pos, cluster.center)
            size = cluster.radius + radius
            pad = 1.0 + 1e-6 * size + 1e-12 * (math.hypot(*pos) + math.hypot(*dr) + travel)
            if first_within(dr, vel, acc, size + pad, dt_max) is None:
                continue
            for rock in cluster.rocks:
                center, rock_radius = self._spheres[rock]
                dt = first_within(_minus(pos, center), vel, acc, radius + rock_radius, dt_max)
                if dt is not None and (best is None or (dt, rock) < best):
                    best = (dt, rock)
        return best
