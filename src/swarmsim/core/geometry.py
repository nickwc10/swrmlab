"""Contact timing and line-of-sight blocking.

``first_within`` finds the first time two bodies, moving under constant relative
acceleration, come within a radius. ``occluded`` decides whether a straight segment
passes through any spherical rock.
"""

import math
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

# A gap at or below this many meters counts as contact.
_TOLERANCE_M = 1e-3
# Safety valve: after this many steps the current time is returned as a hit.
_MAX_STEPS = 100_000


def _vec3(name: str, v: Sequence[float]) -> tuple[float, float, float]:
    if len(v) != 3:
        raise ValueError(f"{name} must have three components")
    x, y, z = float(v[0]), float(v[1]), float(v[2])
    if not (math.isfinite(x) and math.isfinite(y) and math.isfinite(z)):
        raise ValueError(f"{name} must be finite")
    return x, y, z


def _scalar(name: str, v: float) -> float:
    x = float(v)
    if not math.isfinite(x) or x < 0.0:
        raise ValueError(f"{name} must be finite and >= 0")
    return x


def first_within(
    dr: Sequence[float],
    dv: Sequence[float],
    da: Sequence[float],
    radius: float,
    dt_max: float,
) -> float | None:
    """Earliest time in ``[0, dt_max]`` when ``|dr + dv*t + da*t^2/2| <= radius``.

    Returns seconds, or ``None`` if there is no contact in that window. A hit exactly at
    ``dt_max`` counts. Contact is declared when the gap to the sphere is within a
    tolerance of 1e-3 m, so the reported time is never later than the true entry.

    Conservative advancement: with gap ``d`` to the sphere, current relative speed ``s``
    and acceleration magnitude ``A``, the gap cannot close faster than if the whole speed
    and acceleration pointed straight at the target. That takes at least the positive
    root of ``d = s*t + A*t^2/2``, namely ``2d / (s + sqrt(s*s + 2*A*d))``, so stepping
    by that time can never skip past a contact.

    Written in explicit scalar math so a compiled kernel can reproduce it exactly.
    """
    rx, ry, rz = _vec3("dr", dr)
    vx, vy, vz = _vec3("dv", dv)
    ax, ay, az = _vec3("da", da)
    rad = _scalar("radius", radius)
    limit = _scalar("dt_max", dt_max)

    accel = math.sqrt(ax * ax + ay * ay + az * az)
    t = 0.0
    for _ in range(_MAX_STEPS):
        h = 0.5 * t * t
        px = rx + vx * t + ax * h
        py = ry + vy * t + ay * h
        pz = rz + vz * t + az * h
        d = math.sqrt(px * px + py * py + pz * pz) - rad
        if d <= _TOLERANCE_M:
            return t
        if t >= limit:
            return None
        wx = vx + ax * t
        wy = vy + ay * t
        wz = vz + az * t
        s = math.sqrt(wx * wx + wy * wy + wz * wz)
        denom = s + math.sqrt(s * s + 2.0 * accel * d)
        if denom == 0.0:
            return None  # no relative motion and not in contact: never closes
        t = min(t + 2.0 * d / denom, limit)
    return t


@dataclass(frozen=True)
class Rocks:
    """Spherical rocks: centers (M,3), radii (M,) and cluster ids (M,). M may be 0."""

    centers: NDArray[np.float64]
    radii: NDArray[np.float64]
    cluster: NDArray[np.int64]

    def __post_init__(self) -> None:
        centers = np.array(self.centers, dtype=np.float64)
        radii = np.array(self.radii, dtype=np.float64)
        cluster = np.array(self.cluster, dtype=np.int64)
        if centers.ndim != 2 or centers.shape[1] != 3:
            raise ValueError("centers must have shape (M, 3)")
        m = centers.shape[0]
        if radii.shape != (m,):
            raise ValueError("radii must have shape (M,)")
        if cluster.shape != (m,):
            raise ValueError("cluster must have shape (M,)")
        if not np.all(np.isfinite(centers)) or not np.all(np.isfinite(radii)):
            raise ValueError("centers and radii must be finite")
        if np.any(radii <= 0.0):
            raise ValueError("radii must be positive")
        for arr in (centers, radii, cluster):
            arr.flags.writeable = False
        object.__setattr__(self, "centers", centers)
        object.__setattr__(self, "radii", radii)
        object.__setattr__(self, "cluster", cluster)


def occluded(
    a: NDArray[np.float64],
    b: NDArray[np.float64],
    centers: NDArray[np.float64],
    radii: NDArray[np.float64],
) -> bool:
    """True if segment a-b touches any sphere (inclusive; an endpoint inside counts)."""
    if centers.shape[0] == 0:
        return False
    # Work relative to each rock center so precision holds at 3e9 m.
    p = a - centers
    ab = b - a
    length2 = float(ab @ ab)
    if length2 == 0.0:
        s = np.zeros(centers.shape[0])
    else:
        s = np.clip(-(p @ ab) / length2, 0.0, 1.0)
    closest = p + s[:, None] * ab
    dist2 = np.einsum("ij,ij->i", closest, closest)
    return bool(np.any(dist2 <= radii * radii))
