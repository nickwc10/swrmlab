"""Append-only constant-acceleration trajectory segments.

Each unit's path is a list of segments. A segment starts at integer microsecond ``t0``
with position ``pos``, velocity ``vel`` and constant acceleration ``acc``. State at a
later time ``t`` is ``pos + vel*dt + acc*dt^2/2`` with ``dt = (t - t0) / 1e6``. The
difference is taken in int64 microseconds first and converted to seconds second, in this
one place, so every consumer gets identical results.
"""

import operator
from bisect import bisect_right
from collections.abc import Sequence

import numpy as np
from numpy.typing import NDArray

Vec3 = NDArray[np.float64]

_SEGMENT_DTYPE = np.dtype(
    [
        ("unit", "<i4"),
        ("t0", "<i8"),
        ("pos", "<f8", (3,)),
        ("vel", "<f8", (3,)),
        ("acc", "<f8", (3,)),
    ]
)
_INITIAL_CAPACITY = 64
# Times are limited to +-2**62 us so no int64 difference between two times can wrap.
_T_LIMIT = 2**62


def _time(name: str, value: int) -> int:
    t = operator.index(value)
    if not -_T_LIMIT <= t <= _T_LIMIT:
        raise ValueError(f"{name} {t} is outside the supported range +-{_T_LIMIT} us")
    return t


def _vec3(name: str, value: Sequence[float] | NDArray[np.float64]) -> Vec3:
    arr = np.array(value, dtype=np.float64)
    if arr.shape != (3,):
        raise ValueError(f"{name} must have length 3, got shape {arr.shape}")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} must be finite, got {arr}")
    return arr


class SegmentStore:
    def __init__(self) -> None:
        self._rows: NDArray[np.void] = np.zeros(_INITIAL_CAPACITY, dtype=_SEGMENT_DTYPE)
        self._count = 0
        # Per unit: row indices into _rows and the matching t0 values, both ascending.
        self._unit_rows: list[list[int]] = []
        self._unit_t0: list[list[int]] = []
        self._end: list[int | None] = []

    def __len__(self) -> int:
        return len(self._unit_rows)

    def _check_unit(self, unit: int) -> int:
        u = operator.index(unit)
        if u < 0 or u >= len(self._unit_rows):
            raise ValueError(f"unknown unit {unit}")
        return u

    def _append(self, unit: int, t_us: int, pos: Vec3, vel: Vec3, acc: Vec3) -> None:
        if self._count == len(self._rows):
            grown = np.zeros(2 * len(self._rows), dtype=_SEGMENT_DTYPE)
            grown[: self._count] = self._rows[: self._count]
            self._rows = grown
        row = self._rows[self._count]
        row["unit"] = unit
        row["t0"] = t_us
        row["pos"] = pos
        row["vel"] = vel
        row["acc"] = acc
        self._count += 1

    def _eval(self, row_index: int, t_us: int) -> tuple[Vec3, Vec3]:
        row = self._rows[row_index]
        dt = (t_us - int(row["t0"])) / 1e6
        pos = row["pos"] + row["vel"] * dt + row["acc"] * (0.5 * dt * dt)
        vel = row["vel"] + row["acc"] * dt
        return np.array(pos, dtype=np.float64), np.array(vel, dtype=np.float64)

    def add_unit(
        self,
        t_us: int,
        pos: Sequence[float] | NDArray[np.float64],
        vel: Sequence[float] | NDArray[np.float64],
        acc: Sequence[float] | NDArray[np.float64],
    ) -> int:
        t = _time("t_us", t_us)
        p, v, a = _vec3("pos", pos), _vec3("vel", vel), _vec3("acc", acc)
        unit = len(self._unit_rows)
        self._append(unit, t, p, v, a)
        self._unit_rows.append([self._count - 1])
        self._unit_t0.append([t])
        self._end.append(None)
        return unit

    def set_acc(self, unit: int, t_us: int, acc: Sequence[float] | NDArray[np.float64]) -> None:
        u = self._check_unit(unit)
        t = _time("t_us", t_us)
        a = _vec3("acc", acc)
        if self._end[u] is not None:
            raise ValueError(f"unit {u} ended at {self._end[u]}; cannot set acceleration")
        last_t0 = self._unit_t0[u][-1]
        if t < last_t0:
            raise ValueError(f"t_us {t} is before the latest segment start {last_t0}")
        if t == last_t0:
            self._rows[self._unit_rows[u][-1]]["acc"] = a
            return
        pos, vel = self._eval(self._unit_rows[u][-1], t)
        self._append(u, t, pos, vel, a)
        self._unit_rows[u].append(self._count - 1)
        self._unit_t0[u].append(t)

    def end(self, unit: int, t_us: int) -> None:
        u = self._check_unit(unit)
        t = _time("t_us", t_us)
        if self._end[u] is not None:
            raise ValueError(f"unit {u} already ended at {self._end[u]}")
        if t < self._unit_t0[u][-1]:
            raise ValueError(f"end time {t} is before the latest segment start")
        self._end[u] = t

    def birth(self, unit: int) -> int:
        return self._unit_t0[self._check_unit(unit)][0]

    def end_time(self, unit: int) -> int | None:
        return self._end[self._check_unit(unit)]

    def state(self, unit: int, t_us: int) -> tuple[Vec3, Vec3]:
        u = self._check_unit(unit)
        t = _time("t_us", t_us)
        t0s = self._unit_t0[u]
        if t < t0s[0]:
            raise ValueError(f"t_us {t} is before unit {u} was born at {t0s[0]}")
        end = self._end[u]
        if end is not None and t > end:
            raise ValueError(f"t_us {t} is after unit {u} ended at {end}")
        index = bisect_right(t0s, t) - 1
        return self._eval(self._unit_rows[u][index], t)

    def table(self) -> NDArray[np.void]:
        live = self._rows[: self._count]
        order = np.lexsort((live["t0"], live["unit"]))
        return np.array(live[order])
