"""Rocket equation for constant-acceleration burns.

Thrust is adjusted to hold acceleration constant, so total mass falls exponentially:
``m(t) = m0 * exp(-a * dt / ve)``. Masses are total (dry plus fuel), in kilograms.
"""

import math

# No unit may ever move faster than this fraction of c, so light-delay searches hold.
MAX_SPEED_FRACTION = 0.1


def top_speed(speed: float, dry: float, fuel: float, ve: float) -> float:
    """Fastest a unit can get: ``speed`` plus all its delta-v, in m/s."""
    return speed + ve * math.log((dry + fuel) / dry)


def too_fast(top: float, c: float) -> bool:
    """True if a top speed exceeds ``MAX_SPEED_FRACTION`` of ``c``."""
    return top > MAX_SPEED_FRACTION * c


def _require(name: str, value: float, *, positive: bool) -> None:
    if not math.isfinite(value) or value < 0.0 or (positive and value == 0.0):
        kind = "positive" if positive else ">= 0"
        raise ValueError(f"{name} must be finite and {kind}, got {value}")


def fuel_after(mass: float, acc: float, dt_s: float, ve: float) -> float:
    """Total mass left after burning at constant ``acc`` for ``dt_s`` seconds."""
    _require("mass", mass, positive=True)
    _require("acc", acc, positive=False)
    _require("dt_s", dt_s, positive=False)
    _require("ve", ve, positive=True)
    return mass * math.exp(-acc * dt_s / ve)


def burn_time(mass: float, dry: float, acc: float, ve: float) -> float:
    """Seconds of constant ``acc`` until total ``mass`` falls to ``dry``; inf if acc is 0."""
    _require("mass", mass, positive=True)
    _require("dry", dry, positive=True)
    _require("acc", acc, positive=False)
    _require("ve", ve, positive=True)
    if mass < dry:
        raise ValueError(f"mass {mass} is below dry mass {dry}")
    if acc == 0.0:
        return math.inf
    # log1p keeps precision when almost no fuel is left.
    return ve * math.log1p((mass - dry) / dry) / acc
