import copy
import json
import math
import re
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import yaml
from pydantic import BaseModel, ValidationError

from swarmsim.core.engine import World
from swarmsim.core.geometry import Rocks
from swarmsim.core.segments import SegmentStore
from swarmsim.scenario import schema
from swarmsim.scenario.schema import MapSpec, Scenario, ShipClass, content_hash, load

EXAMPLE = Path(__file__).parents[1] / "scenarios" / "two_outposts.yaml"
SHIP = {"dry_mass": 100.0, "fuel_mass": 200.0, "max_acc": 1.0, "ve": 3000.0, "radius": 5.0}
OUTPOST = {"dry_mass": 1e6, "fuel_mass": 0.0, "max_acc": 0.0, "ve": 3000.0, "radius": 50.0}


def raw() -> dict[str, Any]:
    team = {"outpost": OUTPOST, "groups": [{"ship_class": "scout", "count": 2}]}
    return {
        "name": "t",
        "ship_classes": {"scout": dict(SHIP)},
        "blue": copy.deepcopy(team),
        "red": copy.deepcopy(team),
    }


def test_valid_scenario_loads_with_defaults() -> None:
    s = Scenario.model_validate(raw())
    assert s.c == 299792458.0
    assert s.blue.groups[0].initial_speed == 0.0


def test_example_file_loads() -> None:
    s = load(EXAMPLE)
    assert s.map.separation == 3e8
    assert len(s.blue.groups) == 2
    assert sum(g.count for g in s.red.groups) == 4
    assert s.ship_classes["scout"].max_acc == 1.0
    # one 3e8 m crossing at 1 m/s2 needs 2*sqrt(3e8) m/s of delta-v
    for cls in s.ship_classes.values():
        assert cls.ve * math.log(1 + cls.fuel_mass / cls.dry_mass) >= 2 * math.sqrt(3e8)


def test_unknown_field_is_an_error_naming_it() -> None:
    d = raw()
    d["bogus_setting"] = 1
    with pytest.raises(ValidationError, match="bogus_setting"):
        Scenario.model_validate(d)
    d = raw()
    d["ship_classes"]["scout"]["thrust"] = 1.0
    with pytest.raises(ValidationError, match="thrust"):
        Scenario.model_validate(d)


def test_negative_mass_is_an_error_naming_the_field() -> None:
    d = raw()
    d["ship_classes"]["scout"]["dry_mass"] = -1.0
    with pytest.raises(ValidationError, match="dry_mass"):
        Scenario.model_validate(d)
    d = raw()
    d["ship_classes"]["scout"]["fuel_mass"] = -1.0
    with pytest.raises(ValidationError, match="fuel_mass"):
        Scenario.model_validate(d)


@pytest.mark.parametrize("bad", [math.nan, math.inf, -math.inf])
def test_nan_and_infinity_are_rejected(bad: float) -> None:
    d = raw()
    d["ship_classes"]["scout"]["ve"] = bad
    with pytest.raises(ValidationError, match="ve"):
        Scenario.model_validate(d)
    d = raw()
    d["c"] = bad
    with pytest.raises(ValidationError, match="c"):
        Scenario.model_validate(d)


def test_values_above_the_upper_bound_are_rejected() -> None:
    d = raw()
    d["ship_classes"]["scout"]["max_acc"] = 1.1e4
    with pytest.raises(ValidationError, match="max_acc"):
        Scenario.model_validate(d)
    d = raw()
    d["duration"] = 31 * 86400.0
    with pytest.raises(ValidationError, match="duration"):
        Scenario.model_validate(d)
    d = raw()
    d["map"] = {"rock_radius_max": 1.1e5}
    with pytest.raises(ValidationError, match="rock_radius_max"):
        Scenario.model_validate(d)


def test_initial_speed_above_one_percent_of_c_names_the_field() -> None:
    d = raw()
    d["c"] = 1e6
    d["blue"]["groups"][0]["initial_speed"] = 1e4 + 1.0
    with pytest.raises(ValidationError, match=r"blue.*initial_speed"):
        Scenario.model_validate(d)
    d["blue"]["groups"][0]["initial_speed"] = 1e4
    Scenario.model_validate(d)
    d["red"]["groups"][0]["initial_speed"] = 2e4
    with pytest.raises(ValidationError, match=r"red.*initial_speed"):
        Scenario.model_validate(d)


def test_unknown_ship_class_is_named() -> None:
    d = raw()
    d["red"]["groups"][0]["ship_class"] = "dreadnought"
    with pytest.raises(ValidationError, match="dreadnought"):
        Scenario.model_validate(d)


def test_outpost_must_have_no_fuel_and_no_acceleration() -> None:
    d = raw()
    d["blue"]["outpost"]["fuel_mass"] = 1.0
    with pytest.raises(ValidationError, match="fuel_mass"):
        Scenario.model_validate(d)
    d = raw()
    d["blue"]["outpost"]["max_acc"] = 1.0
    with pytest.raises(ValidationError, match="max_acc"):
        Scenario.model_validate(d)


@pytest.mark.parametrize(
    ("over", "field"),
    [
        ({"rock_radius_min": 6e4, "rock_radius_max": 5e4}, "rock_radius_min"),
        ({"home_rocks": 10, "hide_candidates": 11}, "hide_candidates"),
        ({"separation": 1e7, "cluster_radius": 2e6}, "separation"),
        ({"cluster_radius": 4e4, "rock_radius_max": 5e4}, "rock_radius_max"),
    ],
)
def test_inconsistent_map_settings_name_the_field(over: dict[str, float], field: str) -> None:
    with pytest.raises(ValidationError, match=field):
        MapSpec.model_validate(over)


def test_models_are_frozen() -> None:
    ship = ShipClass.model_validate(SHIP)
    with pytest.raises(ValidationError):
        ship.dry_mass = 5.0  # type: ignore[misc]


def test_hash_ignores_key_order_and_explicit_defaults() -> None:
    a = Scenario.model_validate(raw())
    d = raw()
    d["ship_classes"]["scout"] = dict(reversed(list(d["ship_classes"]["scout"].items())))
    d = dict(reversed(list(d.items())))
    d["c"] = 299792458.0  # explicit default
    d["map"] = {"separation": MapSpec().separation}
    b = Scenario.model_validate(d)
    assert content_hash(a) == content_hash(b)
    assert re.fullmatch(r"[0-9a-f]{64}", content_hash(a))


def test_hash_changes_with_any_setting() -> None:
    d = raw()
    d["ship_classes"]["scout"]["fuel_mass"] = 201.0
    assert content_hash(Scenario.model_validate(d)) != content_hash(Scenario.model_validate(raw()))


def test_load_reads_yaml_and_matches_validation(tmp_path: Path) -> None:
    p = tmp_path / "s.yaml"
    p.write_text(yaml.safe_dump(raw()))
    assert load(p) == Scenario.model_validate(raw())


def test_load_errors_are_clear(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="not found"):
        load(tmp_path / "missing.yaml")
    bad = tmp_path / "bad.yaml"
    bad.write_text("a: [unclosed")
    with pytest.raises(ValueError, match="invalid YAML"):
        load(bad)
    listy = tmp_path / "list.yaml"
    listy.write_text("- 1\n- 2\n")
    with pytest.raises(ValueError, match="mapping"):
        load(listy)
    unsafe = tmp_path / "unsafe.yaml"
    unsafe.write_text("!!python/object/apply:os.system ['true']")
    with pytest.raises(ValueError):
        load(unsafe)
    invalid = tmp_path / "invalid.yaml"
    d = raw()
    d["ship_classes"]["scout"]["dry_mass"] = -1
    invalid.write_text(yaml.safe_dump(d))
    with pytest.raises(ValueError, match="dry_mass"):
        load(invalid)  # pydantic's ValidationError is a ValueError


def test_every_numeric_setting_has_default_range_and_description() -> None:
    assert schema.MODELS, "no models registered"
    checked = 0
    for model in schema.MODELS:
        assert issubclass(model, BaseModel)
        for name, field in model.model_fields.items():
            where = f"{model.__name__}.{name}"
            assert field.description, f"{where} has no description"
            if field.annotation not in (int, float):
                continue
            checked += 1
            assert not field.is_required(), f"{where} has no default"
            bounds: dict[str, float] = {}
            for item in field.metadata:
                for key in ("ge", "gt", "le", "lt"):
                    if hasattr(item, key):
                        bounds[key] = getattr(item, key)
            assert bounds.keys() & {"ge", "gt"}, f"{where} has no lower bound"
            assert bounds.keys() & {"le", "lt"}, f"{where} has no upper bound"
            assert all(math.isfinite(v) for v in bounds.values()), f"{where} bound not finite"
            assert re.search(r"\[[^\]]+\]", field.description), f"{where} description lacks units"
            lo = bounds.get("ge", bounds.get("gt"))
            hi = bounds.get("le", bounds.get("lt"))
            assert lo is not None and hi is not None and lo < hi
            default = field.default
            assert lo <= default <= hi, f"{where} default outside its range"
    assert checked >= 20


def test_all_models_are_registered() -> None:
    names = {m.__name__ for m in schema.MODELS}
    assert names == {"ShipClass", "ShipGroup", "TeamSpec", "MapSpec", "Scenario"}
    for m in schema.MODELS:
        assert m.model_config["extra"] == "forbid"
        assert m.model_config["frozen"] is True


def test_hash_is_of_canonical_json() -> None:
    import hashlib

    s = Scenario.model_validate(raw())
    text = json.dumps(s.model_dump(mode="json"), sort_keys=True, separators=(",", ":"))
    assert content_hash(s) == hashlib.sha256(text.encode()).hexdigest()


def test_unflyable_class_names_group_class_and_fields() -> None:
    d = raw()
    d["c"] = 1e3  # 0.1 c = 100 m/s, far below this class's delta-v
    with pytest.raises(ValidationError) as err:
        Scenario.model_validate(d)
    text = str(err.value)
    for word in ("blue.groups[0]", "scout", "top speed", "ve", "fuel_mass", "dry_mass"):
        assert word in text
    d = raw()
    d["ship_classes"]["scout"].update(ve=3e7, fuel_mass=1e5)
    with pytest.raises(ValidationError, match="top speed"):
        Scenario.model_validate(d)


def test_dry_mass_has_a_realistic_floor() -> None:
    d = raw()
    d["ship_classes"]["scout"]["dry_mass"] = 5e-324
    with pytest.raises(ValidationError, match="dry_mass"):
        Scenario.model_validate(d)


def test_load_and_world_agree_on_the_speed_limit_exactly() -> None:
    c, ve, dry, fuel = 1e6, 30_000.0, 100.0, 2000.0
    edge = 0.1 * c - ve * math.log((dry + fuel) / dry)
    assert 0.0 < edge < 0.01 * c
    # steps of a few ulps of the sum (about 1.5e-11) on both sides of the edge
    speeds = [edge + k * 1e-11 for k in range(-6, 7)]
    verdicts = set()
    for speed in speeds:
        d = raw()
        d["c"] = c
        d["ship_classes"]["scout"] = {**SHIP, "dry_mass": dry, "fuel_mass": fuel, "ve": ve}
        d["blue"]["groups"][0]["initial_speed"] = float(speed)
        try:
            Scenario.model_validate(d)
            loads = True
        except ValidationError:
            loads = False
        world = World(
            SegmentStore(), Rocks(np.zeros((0, 3)), np.zeros(0), np.zeros(0)), c, horizon_us=1
        )
        try:
            world.spawn(
                0, (0.0, 0.0, 0.0), (float(speed), 0.0, 0.0), team="b", cls="scout",
                dry_mass=dry, fuel_mass=fuel, max_acc=1.0, ve=ve, radius=5.0,
            )  # fmt: skip
            spawns = True
        except ValueError:
            spawns = False
        assert loads == spawns, speed
        verdicts.add(loads)
    assert verdicts == {True, False}  # the speeds straddle the limit


def test_negative_zero_hashes_like_zero() -> None:
    d = raw()
    d["blue"]["groups"][0]["initial_speed"] = -0.0
    a, b = Scenario.model_validate(d), Scenario.model_validate(raw())
    assert a == b
    assert content_hash(a) == content_hash(b)
    assert math.copysign(1.0, a.blue.groups[0].initial_speed) == 1.0
