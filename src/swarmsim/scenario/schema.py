"""Scenario settings: validated models, YAML loading and a content hash.

Every number is a field with a default, finite bounds and a description that gives its
unit in square brackets. The defaults and bounds are design choices, not sourced data.
Models are frozen and reject unknown fields, NaN and infinity, so a typo or an absurd
value fails when the file loads instead of changing a result silently.
"""

import hashlib
import json
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from swarmsim.core.propulsion import MAX_SPEED_FRACTION, too_fast, top_speed

# Ships may start at no more than this fraction of c, so light delay stays well defined.
_MAX_START_SPEED_FRACTION = 0.01
_DAY_S = 86_400.0
_MODEL_CONFIG = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)


class _Model(BaseModel):
    model_config = _MODEL_CONFIG

    @field_validator("*")
    @classmethod
    def _no_negative_zero(cls, value: Any) -> Any:
        # -0.0 equals 0.0 but would hash differently; adding 0.0 turns it into +0.0.
        return value + 0.0 if isinstance(value, float) else value


def _num(default: float, lo: float, hi: float, description: str, *, open_lo: bool = False) -> Any:
    """A float field bounded to [lo, hi], or (lo, hi] when ``open_lo``."""
    if open_lo:
        return Field(default, gt=lo, le=hi, description=description)
    return Field(default, ge=lo, le=hi, description=description)


def _count(default: int, lo: int, hi: int, description: str) -> Any:
    return Field(default, ge=lo, le=hi, description=description)


class ShipClass(_Model):
    """Physical limits of one kind of ship (also used for outposts)."""

    dry_mass: float = _num(1000.0, 1.0, 1e12, "Mass without fuel, at least 1 kg [kg]")
    fuel_mass: float = _num(1000.0, 0.0, 1e12, "Fuel mass at the start [kg]")
    max_acc: float = _num(1.0, 0.0, 1e4, "Largest acceleration the engine can give [m/s2]")
    ve: float = _num(30_000.0, 0.0, 3e7, "Exhaust speed of the engine [m/s]", open_lo=True)
    radius: float = _num(10.0, 0.0, 1e5, "Radius of the ship as a sphere, used for contact [m]")


class ShipGroup(_Model):
    """``count`` identical ships of one class that start together."""

    ship_class: str = Field(description="Name of an entry in the scenario's ship_classes [name]")
    count: int = _count(1, 1, 1000, "Number of ships in this group [ships]")
    initial_speed: float = _num(
        0.0,
        0.0,
        3e6,
        "Starting speed toward the enemy along the x axis, at most 0.01 c [m/s]",
    )


class TeamSpec(_Model):
    """One team: its outpost and the ships it starts with."""

    outpost: ShipClass = Field(
        default=ShipClass(dry_mass=1e6, fuel_mass=0.0, max_acc=0.0, ve=30_000.0, radius=100.0),
        description="Outpost body; fuel_mass and max_acc must be 0 because it never moves",
    )
    groups: tuple[ShipGroup, ...] = Field(
        default=(), max_length=20, description="Ship groups, spawned in this order"
    )

    @model_validator(mode="after")
    def _outpost_is_inert(self) -> "TeamSpec":
        for name in ("fuel_mass", "max_acc"):
            if getattr(self.outpost, name) != 0.0:
                raise ValueError(f"outpost.{name} must be 0 because an outpost never moves")
        return self


class MapSpec(_Model):
    """Asteroid field. Rocks are made in the west half and mirrored to the east."""

    separation: float = _num(3e8, 1e3, 1e12, "Distance between the two home clusters [m]")
    cluster_radius: float = _num(2e6, 1.0, 1e10, "Radius of every rock cluster [m]")
    rock_radius_min: float = _num(1e3, 1.0, 1e5, "Smallest rock radius [m]")
    rock_radius_max: float = _num(5e4, 1.0, 1e5, "Largest rock radius [m]")
    home_rocks: int = _count(60, 1, 5000, "Rocks in each home cluster [rocks]")
    hide_candidates: int = _count(
        40, 1, 5000, "Outposts hide on one of this many first rocks of the home cluster [rocks]"
    )
    mid_clusters: int = _count(
        2, 0, 100, "Clusters per side between its home cluster and the middle [clusters]"
    )
    mid_rocks: int = _count(30, 0, 5000, "Rocks in each of those clusters [rocks]")
    mid_lateral: float = _num(
        1e7, 0.0, 1e11, "Those clusters are centered within this far of the x axis [m]"
    )
    outpost_clearance: float = _num(
        100.0, 1.0, 1e6, "Gap between an outpost or ship and any rock surface at the start [m]"
    )
    ship_spread: float = _num(
        2e4, 0.0, 1e9, "Ships start within this much farther from their outpost than touching [m]"
    )

    @model_validator(mode="after")
    def _consistent(self) -> "MapSpec":
        if self.rock_radius_min > self.rock_radius_max:
            raise ValueError("rock_radius_min must not exceed rock_radius_max")
        if self.rock_radius_max > self.cluster_radius:
            raise ValueError("rock_radius_max must not exceed cluster_radius")
        if self.hide_candidates > self.home_rocks:
            raise ValueError("hide_candidates must not exceed home_rocks")
        # Room for the home cluster, the middle clusters and the gap before the mirror line.
        if self.separation <= 8.0 * self.cluster_radius:
            raise ValueError("separation must exceed 8 x cluster_radius to leave room for clusters")
        return self


class Scenario(_Model):
    """A complete battle setup."""

    name: str = Field(description="Short label for the scenario [name]")
    c: float = _num(299_792_458.0, 1e3, 1e12, "Speed of light [m/s]")
    duration: float = _num(172_800.0, 1.0, 30.0 * _DAY_S, "Length of the battle [s]")
    map: MapSpec = Field(default=MapSpec(), description="Asteroid field settings")
    ship_classes: dict[str, ShipClass] = Field(description="Ship classes by name")
    blue: TeamSpec = Field(description="The blue team")
    red: TeamSpec = Field(description="The red team")

    @model_validator(mode="after")
    def _groups_are_valid(self) -> "Scenario":
        limit = _MAX_START_SPEED_FRACTION * self.c
        for team_name, team in (("blue", self.blue), ("red", self.red)):
            for i, group in enumerate(team.groups):
                where = f"{team_name}.groups[{i}]"
                if group.ship_class not in self.ship_classes:
                    known = ", ".join(sorted(self.ship_classes)) or "none"
                    raise ValueError(
                        f"{where}.ship_class: unknown ship class '{group.ship_class}' "
                        f"(known: {known})"
                    )
                if group.initial_speed > limit:
                    raise ValueError(
                        f"{where}.initial_speed {group.initial_speed} m/s exceeds "
                        f"0.01 c = {limit} m/s"
                    )
                cls = self.ship_classes[group.ship_class]
                top = top_speed(group.initial_speed, cls.dry_mass, cls.fuel_mass, cls.ve)
                if too_fast(top, self.c):
                    raise ValueError(
                        f"{where}: ship class '{group.ship_class}' has top speed {top} m/s, "
                        f"above {MAX_SPEED_FRACTION} c = {MAX_SPEED_FRACTION * self.c} m/s; "
                        "reduce ve or fuel_mass, raise dry_mass, or lower initial_speed"
                    )
        return self


MODELS: tuple[type[BaseModel], ...] = (ShipClass, ShipGroup, TeamSpec, MapSpec, Scenario)


def load(path: str | Path) -> Scenario:
    """Read a scenario YAML file and validate it."""
    p = Path(path)
    try:
        text = p.read_text(encoding="utf-8")
    except FileNotFoundError:
        raise ValueError(f"scenario file not found: {p}") from None
    except OSError as err:
        raise ValueError(f"cannot read scenario file {p}: {err}") from None
    try:
        data = yaml.safe_load(text)
    except yaml.YAMLError as err:
        raise ValueError(f"invalid YAML in {p}: {err}") from None
    if not isinstance(data, dict):
        raise ValueError(f"{p} must contain a mapping of settings at the top level")
    return Scenario.model_validate(data)


def content_hash(scenario: Scenario) -> str:
    """SHA-256 of the validated scenario as canonical JSON (sorted keys)."""
    text = json.dumps(scenario.model_dump(mode="json"), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(text.encode("utf-8")).hexdigest()
