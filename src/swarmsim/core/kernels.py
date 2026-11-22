"""Compiled batch versions of ``light.emission_time`` and ``geometry.first_within``.

Same rules and the same arithmetic in the same order as the reference functions, run on
plain arrays by Numba on one thread, without fastmath (which would let the compiler
reorder sums and change answers). Emission times are bisected through the same
microseconds as the reference, so they are the same integers. Contact times follow the
same steps and agree within 1e-12 relative.

The first call in a process compiles the kernel, which takes under a second.
"""

import math
from collections.abc import Callable
from dataclasses import dataclass
from typing import cast

import numpy as np
from numba import njit
from numpy.typing import ArrayLike, NDArray

from swarmsim.core.geometry import _MAX_STEPS, _TOLERANCE_M
from swarmsim.core.light import _T_LIMIT, C, _check_c
from swarmsim.core.segments import SegmentStore

# Both sentinels lie outside the supported time range of +-2**61 us.
NOT_SEEN: int = -(2**63)  # emission_time_batch: the reference returns None
NO_END: int = 2**63 - 1  # Paths.end: the unit has not ended


def _compiled[F: Callable[..., object]](f: F) -> F:
    """Compile ``f``; typed here because Numba itself is untyped."""
    return cast(F, njit(fastmath=False, parallel=False)(f))


@dataclass(frozen=True)
class Paths:
    """Every unit's trajectory as plain arrays. Unit u owns rows starts[u]:starts[u+1]."""

    starts: NDArray[np.int64]  # (units + 1,)
    t0: NDArray[np.int64]  # (segments,), ascending within a unit
    motion: NDArray[np.float64]  # (segments, 9): position, velocity, acceleration
    birth: NDArray[np.int64]  # (units,)
    end: NDArray[np.int64]  # (units,), NO_END where the unit has not ended


def pack(store: SegmentStore) -> Paths:
    """Copy a store into arrays. Pack again after the store changes."""
    table = store.table()
    n = len(store)
    ends = [store.end_time(u) for u in range(n)]
    return Paths(
        starts=np.searchsorted(table["unit"], np.arange(n + 1)).astype(np.int64),
        t0=table["t0"].astype(np.int64),
        motion=np.concatenate([table["pos"], table["vel"], table["acc"]], axis=1),
        birth=np.array([store.birth(u) for u in range(n)], dtype=np.int64),
        end=np.array([NO_END if e is None else e for e in ends], dtype=np.int64),
    )


@_compiled
def _segment(t0: NDArray[np.int64], a: int, b: int, t: int) -> int:
    """Last row in a..b that starts at or before ``t``; row a must."""
    while a < b:
        mid = (a + b + 1) // 2
        if t0[mid] <= t:
            a = mid
        else:
            b = mid - 1
    return a


@_compiled
def _late(
    t0: NDArray[np.int64],
    m: NDArray[np.float64],
    k: int,
    t: int,
    t_obs: int,
    px: float,
    py: float,
    pz: float,
    c: float,
) -> float:
    """Distance at ``t`` (on row k) minus how far light gets by ``t_obs``; <= 0 is seen."""
    dt = (t - t0[k]) / 1e6
    h = 0.5 * dt * dt
    dx = (m[k, 0] + m[k, 3] * dt + m[k, 6] * h) - px
    dy = (m[k, 1] + m[k, 4] * dt + m[k, 7] * h) - py
    dz = (m[k, 2] + m[k, 5] * dt + m[k, 8] * h) - pz
    return math.sqrt(dx * dx + dy * dy + dz * dz) - c * ((t_obs - t) / 1e6)


@_compiled
def _emission_times(
    starts: NDArray[np.int64],
    t0: NDArray[np.int64],
    m: NDArray[np.float64],
    birth: NDArray[np.int64],
    end: NDArray[np.int64],
    unit: NDArray[np.int64],
    t_obs: NDArray[np.int64],
    pos: NDArray[np.float64],
    c: float,
    out: NDArray[np.int64],
) -> None:
    for i in range(unit.shape[0]):
        u, t = unit[i], t_obs[i]
        px, py, pz = pos[i, 0], pos[i, 1], pos[i, 2]
        lo, hi = birth[u], min(t, end[u])
        out[i] = NOT_SEEN
        if t < lo:
            continue
        a = starts[u]
        b = _segment(t0, a, starts[u + 1] - 1, hi)
        f_hi = _late(t0, m, b, hi, t, px, py, pz, c)
        if end[u] <= t and f_hi <= 0.0:
            continue
        if _late(t0, m, a, lo, t, px, py, pz, c) > 0.0:
            continue
        if f_hi <= 0.0:
            out[i] = hi
            continue
        # The answer is in lo..hi, on one of rows a..b; both ranges shrink together.
        while hi - lo > 1:
            mid = lo + (hi - lo) // 2
            k = _segment(t0, a, b, mid)
            if _late(t0, m, k, mid, t, px, py, pz, c) <= 0.0:
                lo, a = mid, k
            else:
                hi, b = mid, k
        out[i] = lo


@_compiled
def _contact_times(
    dr: NDArray[np.float64],
    dv: NDArray[np.float64],
    da: NDArray[np.float64],
    radius: NDArray[np.float64],
    dt_max: NDArray[np.float64],
    out: NDArray[np.float64],
) -> None:
    for i in range(radius.shape[0]):
        rx, ry, rz = dr[i, 0], dr[i, 1], dr[i, 2]
        vx, vy, vz = dv[i, 0], dv[i, 1], dv[i, 2]
        ax, ay, az = da[i, 0], da[i, 1], da[i, 2]
        rad, limit = radius[i], dt_max[i]
        accel = math.sqrt(ax * ax + ay * ay + az * az)
        t = 0.0
        result = math.nan
        for _ in range(_MAX_STEPS):
            h = 0.5 * t * t
            px = rx + vx * t + ax * h
            py = ry + vy * t + ay * h
            pz = rz + vz * t + az * h
            d = math.sqrt(px * px + py * py + pz * pz) - rad
            if d <= _TOLERANCE_M:
                result = t
                break
            if t >= limit:
                break
            wx = vx + ax * t
            wy = vy + ay * t
            wz = vz + az * t
            s = math.sqrt(wx * wx + wy * wy + wz * wz)
            denom = s + math.sqrt(s * s + 2.0 * accel * d)
            if denom == 0.0:
                break
            t = min(t + 2.0 * d / denom, limit)
        else:
            result = t  # the step cap counts as a hit, as in the reference
        out[i] = result


def _ints(name: str, value: ArrayLike, n: int) -> NDArray[np.int64]:
    arr = np.asarray(value)
    if arr.size and arr.dtype.kind not in "iu":
        raise ValueError(f"{name} must be integers, got {arr.dtype}")
    if arr.shape != (n,):
        raise ValueError(f"{name} must have shape ({n},), got {arr.shape}")
    return np.ascontiguousarray(arr, dtype=np.int64)


def _floats(name: str, value: ArrayLike, shape: tuple[int, ...]) -> NDArray[np.float64]:
    arr = np.ascontiguousarray(value, dtype=np.float64)
    if arr.shape != shape:
        raise ValueError(f"{name} must have shape {shape}, got {arr.shape}")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} must be finite")
    return arr


def emission_time_batch(
    paths: Paths, unit: ArrayLike, t_obs_us: ArrayLike, obs_pos: ArrayLike, c: float = C
) -> NDArray[np.int64]:
    """``light.emission_time`` for N queries: ``unit`` (N,), ``t_obs_us`` (N,), ``obs_pos`` (N,3).

    Returns int64 times, with ``NOT_SEEN`` wherever the reference returns None.
    """
    c = _check_c(c)
    n = np.asarray(unit).size
    units, t_obs = _ints("unit", unit, n), _ints("t_obs_us", t_obs_us, n)
    pos = _floats("obs_pos", obs_pos, (n, 3))
    if n and (units.min() < 0 or units.max() >= len(paths.birth)):
        raise ValueError("unit ids must exist in paths")
    if n and (t_obs.min() < -_T_LIMIT or t_obs.max() > _T_LIMIT):
        raise ValueError(f"t_obs_us is outside the supported range +-{_T_LIMIT} us")
    out = np.empty(n, dtype=np.int64)
    _emission_times(
        paths.starts, paths.t0, paths.motion, paths.birth, paths.end, units, t_obs, pos, c, out
    )
    return out


def first_within_batch(
    dr: ArrayLike, dv: ArrayLike, da: ArrayLike, radius: ArrayLike, dt_max: ArrayLike
) -> NDArray[np.float64]:
    """``geometry.first_within`` for N pairs: ``dr``, ``dv``, ``da`` (N,3), the rest (N,).

    Returns seconds, with NaN wherever the reference returns None.
    """
    n = np.asarray(radius).size
    r, v, a = (_floats(k, x, (n, 3)) for k, x in (("dr", dr), ("dv", dv), ("da", da)))
    rad, limit = _floats("radius", radius, (n,)), _floats("dt_max", dt_max, (n,))
    for name, arr in (("radius", rad), ("dt_max", limit)):
        if n and arr.min() < 0.0:
            raise ValueError(f"{name} must be >= 0")
    out = np.empty(n, dtype=np.float64)
    _contact_times(r, v, a, rad, limit, out)
    return out
