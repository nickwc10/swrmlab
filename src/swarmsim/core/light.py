"""Light delay: where an observer sees a unit, and when a message reaches a receiver.

Both functions bisect on integer microseconds over a :class:`SegmentStore` trajectory.
The caller guarantees every unit moves slower than ``c``; this is not checked. Under that
guarantee the light-cone functions below are monotonic in time, including across segment
boundaries (position is continuous), so bisection is exact. Answers are tight: the
returned time satisfies the light condition and the neighbouring microsecond does not,
so light may be up to 1 us late and is never early.

Distance is computed with explicit scalar math so compiled kernels can reproduce it
bit for bit. Time differences are taken in integer microseconds, then divided by 1e6.
"""

import math
import operator
from collections.abc import Sequence

import numpy as np
from numpy.typing import NDArray

from swarmsim.core.segments import SegmentStore

C: float = 299792458.0
_T_LIMIT = 2**61


def _check_c(c: float) -> float:
    c = float(c)
    if not math.isfinite(c) or c <= 0.0:
        raise ValueError(f"c must be finite and positive, got {c}")
    return c


def _point(name: str, value: Sequence[float] | NDArray[np.float64]) -> tuple[float, float, float]:
    arr = np.array(value, dtype=np.float64)
    if arr.shape != (3,) or not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} must be three finite values, got {value}")
    return float(arr[0]), float(arr[1]), float(arr[2])


def _time(name: str, value: int) -> int:
    t = operator.index(value)
    if not -_T_LIMIT <= t <= _T_LIMIT:
        raise ValueError(f"{name} {t} is outside the supported range +-{_T_LIMIT} us")
    return t


def _distance(store: SegmentStore, unit: int, t_us: int, p: tuple[float, float, float]) -> float:
    pos, _ = store.state(unit, t_us)
    dx = float(pos[0]) - p[0]
    dy = float(pos[1]) - p[1]
    dz = float(pos[2]) - p[2]
    return math.sqrt(dx * dx + dy * dy + dz * dz)


def emission_time(
    store: SegmentStore,
    unit: int,
    t_obs_us: int,
    obs_pos: Sequence[float] | NDArray[np.float64],
    c: float = C,
) -> int | None:
    """Time at which ``unit`` emitted the light that reaches ``obs_pos`` at ``t_obs_us``.

    Returns the largest t in [birth, min(t_obs_us, end)] with
    ``distance(x(t), obs_pos) <= c * (t_obs_us - t) / 1e6``. Returns None if the
    observation precedes the unit's birth, if first light has not yet arrived, or if the
    unit has ended and the light of its end has already arrived (it is no longer seen).
    The unit's speed must stay below ``c`` (not checked).
    """
    c = _check_c(c)
    p = _point("obs_pos", obs_pos)
    t_obs = _time("t_obs_us", t_obs_us)
    birth = store.birth(unit)
    end = store.end_time(unit)
    if t_obs < birth:
        return None

    def f(t: int) -> float:
        return _distance(store, unit, t, p) - c * ((t_obs - t) / 1e6)

    hi = t_obs if end is None else min(t_obs, end)
    if end is not None and end <= t_obs and f(end) <= 0.0:
        return None
    lo = birth
    if f(lo) > 0.0:
        return None
    if f(hi) <= 0.0:
        return hi
    # Invariant: f(lo) <= 0 < f(hi).
    while hi - lo > 1:
        mid = lo + (hi - lo) // 2
        if f(mid) <= 0.0:
            lo = mid
        else:
            hi = mid
    return lo


def arrival_time(
    store: SegmentStore,
    receiver: int,
    t_send_us: int,
    send_pos: Sequence[float] | NDArray[np.float64],
    horizon_us: int,
    c: float = C,
) -> int | None:
    """Time at which a message sent from ``send_pos`` at ``t_send_us`` reaches ``receiver``.

    Returns the smallest t in [max(t_send_us, birth), min(horizon_us, end)] with
    ``distance(x(t), send_pos) <= c * (t - t_send_us) / 1e6``, or None if the message has
    not arrived by then. A message is a wavefront that passes each point once: a receiver
    born after the wavefront has passed its birth position never gets that message (None);
    exactly at birth it receives it at birth. Valid only for the receiver's trajectory as
    currently planned (the last segment is extrapolated); callers recompute after the next
    ``set_acc``. The receiver's speed must stay below ``c`` (not checked).

    The receiver's current segment is extrapolated past fuel-out, so a burning receiver's
    imagined speed can exceed ``c`` even though its real speed cannot. The search may then
    return None for a message that really arrives before fuel-out, and recomputing after
    fuel-out does not recover it. Callers must therefore limit ``horizon_us`` to the
    receiver's predicted fuel-out and repeat at each segment change.
    """
    c = _check_c(c)
    p = _point("send_pos", send_pos)
    t_send = _time("t_send_us", t_send_us)
    horizon = _time("horizon_us", horizon_us)
    end = store.end_time(receiver)
    lo = max(t_send, store.birth(receiver))
    hi = horizon if end is None else min(horizon, end)
    if lo > hi:
        return None

    def g(t: int) -> float:
        return _distance(store, receiver, t, p) - c * ((t - t_send) / 1e6)

    g_lo = g(lo)
    if t_send < store.birth(receiver):
        # The wavefront passes each point once. If it had already passed the birth
        # position before the receiver existed, the receiver never gets this message.
        if g_lo < 0.0:
            return None
        if g_lo == 0.0:
            return lo
    if g_lo <= 0.0:
        return lo
    if g(hi) > 0.0:
        return None
    # Invariant: g(lo) > 0 >= g(hi).
    while hi - lo > 1:
        mid = lo + (hi - lo) // 2
        if g(mid) <= 0.0:
            hi = mid
        else:
            lo = mid
    return hi
