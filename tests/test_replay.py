import hashlib
import json
import math
import os
import struct
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from swarmsim import SIM_VERSION
from swarmsim.core.engine import World
from swarmsim.replay import Replay, ReplayError, read, write
from swarmsim.scenario.build import build_world
from swarmsim.scenario.schema import Scenario, content_hash, load

EXAMPLE = Path(__file__).parents[1] / "scenarios" / "two_outposts.yaml"

SMALL_MAP = {
    "separation": 1e8,
    "cluster_radius": 1e6,
    "rock_radius_max": 2e4,
    "home_rocks": 20,
    "hide_candidates": 10,
    "mid_clusters": 1,
    "mid_rocks": 10,
}


def small_scenario(duration: float = 3600.0) -> Scenario:
    team = {
        "outpost": {
            "dry_mass": 1e6,
            "fuel_mass": 0.0,
            "max_acc": 0.0,
            "ve": 3000.0,
            "radius": 50.0,
        },
        "groups": [{"ship_class": "scout", "count": 2, "initial_speed": 100.0}],
    }
    classes = {
        "scout": {"dry_mass": 100.0, "fuel_mass": 500.0, "max_acc": 1.0, "ve": 3e4, "radius": 5.0},
    }
    return Scenario.model_validate(
        {
            "name": "t",
            "duration": duration,
            "map": SMALL_MAP,
            "ship_classes": classes,
            "blue": team,
            "red": team,
        }
    )


def burn_all(world: World, burns: int, seed: int = 0) -> None:
    """Give every ship ``burns`` burns spread over the run, then run to the horizon."""
    rng = np.random.default_rng(seed)
    for t in np.linspace(0, world.horizon_us, burns + 2)[1:-1]:
        world.run_until(int(t))
        for unit, _team, cls, _birth, end in world.units:
            if cls != "outpost" and end is None:
                d = rng.normal(size=3)
                world.command_acc(unit, int(t), d / np.linalg.norm(d) * rng.uniform(0.1, 1.0))
    world.run_until(world.horizon_us)


def make_run(
    seed: int = 1, burns: int = 6, scenario: Scenario | None = None, swap: bool = False
) -> tuple[World, Scenario]:
    s = scenario or small_scenario()
    w = build_world(s, seed, swap)
    burn_all(w, burns)
    return w, s


def split(data: bytes) -> tuple[dict[str, Any], bytes]:
    (n,) = struct.unpack("<I", data[4:8])
    return json.loads(data[8 : 8 + n]), data[8 + n :]


def join(header: dict[str, Any], body: bytes, magic: bytes = b"SWR2") -> bytes:
    h = json.dumps(header, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    return magic + struct.pack("<I", len(h)) + h + body


@pytest.fixture(scope="module")
def good(tmp_path_factory: pytest.TempPathFactory) -> tuple[bytes, World, Scenario]:
    w, s = make_run()
    p = tmp_path_factory.mktemp("good") / "a.swr"
    write(p, w, s, 1)
    return p.read_bytes(), w, s


def read_bytes(tmp_path: Path, data: bytes) -> Replay:
    p = tmp_path / "x.swr"
    p.write_bytes(data)
    return read(p)


def test_same_run_gives_the_same_bytes(tmp_path: Path) -> None:
    digests = []
    for i in range(2):
        w, s = make_run(seed=5)
        p = tmp_path / f"{i}.swr"
        write(p, w, s, 5)
        digests.append(hashlib.sha256(p.read_bytes()).hexdigest())
    assert digests[0] == digests[1]
    w, s = make_run(seed=6)
    write(tmp_path / "other.swr", w, s, 6)
    assert hashlib.sha256((tmp_path / "other.swr").read_bytes()).hexdigest() != digests[0]


def test_file_starts_with_magic_and_header_is_canonical_json(
    good: tuple[bytes, World, Scenario],
) -> None:
    data = good[0]
    assert data[:4] == b"SWR2"
    (n,) = struct.unpack("<I", data[4:8])
    raw = data[8 : 8 + n]
    header = json.loads(raw)
    assert raw == json.dumps(
        header, sort_keys=True, separators=(",", ":"), allow_nan=False, ensure_ascii=False
    ).encode("utf-8")
    assert header["version"] == 2
    assert header["sim_version"] == SIM_VERSION
    assert [s["name"] for s in header["sections"]] == ["segments"]


def test_header_identifies_and_can_rebuild_the_run(
    tmp_path: Path,
) -> None:
    s = small_scenario()
    w, _ = make_run(seed=9, scenario=s, swap=True)
    write(tmp_path / "a.swr", w, s, 9, swap=True)
    r = read(tmp_path / "a.swr")
    h = r.header
    assert h["seed"] == 9 and h["swap"] is True
    assert h["horizon_us"] == w.horizon_us
    assert h["scenario_hash"] == content_hash(s)
    assert r.scenario == s
    assert [(u["id"], u["team"], u["cls"], u["birth"], u["end"]) for u in h["units"]] == [
        tuple(u) for u in w.units
    ]
    rebuilt = build_world(r.scenario, h["seed"], h["swap"])
    for unit, *_ in rebuilt.units:
        a, b = rebuilt.store.state(unit, 0), r.state(unit, 0)
        assert a[0].tobytes() == b[0].tobytes() and a[1].tobytes() == b[1].tobytes()
    assert np.array_equal(r.rocks.centers, w.rocks.centers)
    assert np.array_equal(r.rocks.radii, w.rocks.radii)
    assert np.array_equal(r.rocks.cluster, w.rocks.cluster)


def test_events_round_trip(good: tuple[bytes, World, Scenario], tmp_path: Path) -> None:
    data, w, _ = good
    r = read_bytes(tmp_path, data)
    assert len(w.events) > 10
    assert list(r.events) == w.events


def test_state_equals_the_store_exactly_at_100_times(
    good: tuple[bytes, World, Scenario], tmp_path: Path
) -> None:
    data, w, _ = good
    r = read_bytes(tmp_path, data)
    rng = np.random.default_rng(3)
    checked = 0
    for unit, _team, _cls, birth, end in w.units:
        last = w.horizon_us if end is None else end
        times = [birth, last, *rng.integers(birth, last + 1, size=20).tolist()]
        if end is None:
            times.append(w.horizon_us + 12_345_678)  # a live unit extrapolates
        for t in times:
            ps, vs = w.store.state(unit, t)
            pr, vr = r.state(unit, t)
            assert ps.tobytes() == pr.tobytes() and vs.tobytes() == vr.tobytes()
            checked += 1
    assert checked >= 100


def test_state_rejects_times_and_units_outside_a_units_life(tmp_path: Path) -> None:
    s = small_scenario()
    w = build_world(s, 1)
    w.spawn(
        2_000_000, (1e5, 0, 0), (0, 0, 0), team="blue", cls="scout", dry_mass=10.0,
        fuel_mass=0.0, max_acc=0.0, ve=1000.0, radius=1.0,
    )  # fmt: skip
    late = len(w.units) - 1
    w.run_until(1_000_000)  # stops before the late unit is born
    first = 1
    w.command_acc(first, 1_000_000, (1.0, 0.0, 0.0))
    w.run_until(w.horizon_us)
    write(tmp_path / "a.swr", w, s, 1)
    r = read(tmp_path / "a.swr")
    with pytest.raises(ValueError, match="before"):
        r.state(late, 1_999_999)
    assert r.state(late, 2_000_000)[0].tolist() == [1e5, 0.0, 0.0]
    with pytest.raises(ValueError, match="unknown unit"):
        r.state(len(w.units), 0)
    with pytest.raises(ValueError, match="unknown unit"):
        r.state(-1, 0)
    # Find any unit that ended and check the after-end error matches the store.
    s2 = small_scenario(duration=86_400.0)
    w2, _ = make_run(seed=2, scenario=s2, burns=40)
    write(tmp_path / "b.swr", w2, s2, 2)
    r2 = read(tmp_path / "b.swr")
    ended = [u for u in w2.units if u[4] is not None]
    assert ended, "scenario should lose at least one unit to a rock"
    u = ended[0]
    with pytest.raises(ValueError, match="after"):
        r2.state(u[0], u[4] + 1)
    with pytest.raises(ValueError, match="before"):
        r2.state(u[0], u[3] - 1)
    assert r2.state(u[0], u[4])[0].tobytes() == w2.store.state(u[0], u[4])[0].tobytes()


def test_run_stopped_early_keeps_the_header_horizon(tmp_path: Path) -> None:
    s = small_scenario()
    w = build_world(s, 4)
    w.run_until(1_000_000)
    write(tmp_path / "a.swr", w, s, 4)
    r = read(tmp_path / "a.swr")
    assert r.header["horizon_us"] == w.horizon_us
    assert r.state(1, 5_000_000)[0].shape == (3,)  # extrapolates the last segment


def test_returned_arrays_are_copies(good: tuple[bytes, World, Scenario], tmp_path: Path) -> None:
    r = read_bytes(tmp_path, good[0])
    p, v = r.state(1, 0)
    expected = p.copy()
    p[:] = 9.9
    v[:] = 9.9
    assert np.array_equal(r.state(1, 0)[0], expected)
    assert not r.rocks.centers.flags.writeable


def test_a_missing_file_raises_file_not_found(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        read(tmp_path / "nope.swr")


def test_write_is_atomic(
    good: tuple[bytes, World, Scenario], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _, w, s = good
    target = tmp_path / "a.swr"

    def boom(*_: object, **__: object) -> None:
        raise OSError("disk gone")

    with monkeypatch.context() as m:
        m.setattr(os, "replace", boom)
        with pytest.raises(OSError, match="disk gone"):
            write(target, w, s, 1)
    assert list(tmp_path.iterdir()) == []
    write(target, w, s, 1)
    before = target.read_bytes()
    with monkeypatch.context() as m:
        m.setattr(os, "replace", boom)
        with pytest.raises(OSError):
            write(target, w, s, 1)
    assert target.read_bytes() == before
    assert [p.name for p in tmp_path.iterdir()] == ["a.swr"]


def test_write_failing_to_encode_leaves_nothing(tmp_path: Path) -> None:
    w, s = make_run()
    w.events.append(type(w.events[0])(0, "bad", 0, None, {"x": math.nan}))
    with pytest.raises(ValueError):
        write(tmp_path / "a.swr", w, s, 1)
    assert list(tmp_path.iterdir()) == []


# ---- malformed files ----


def test_truncation_at_any_length_raises_replay_error(tmp_path: Path) -> None:
    w, s = make_run(burns=2)
    p = tmp_path / "a.swr"
    write(p, w, s, 1)
    data = p.read_bytes()
    header_end = 8 + struct.unpack("<I", data[4:8])[0]
    rng = np.random.default_rng(0)
    lengths = set(range(0, 40))
    lengths |= set(range(header_end - 5, min(header_end + 200, len(data))))
    lengths |= set(rng.integers(0, len(data), size=300).tolist())
    lengths.add(len(data) - 1)
    for n in sorted(lengths):
        with pytest.raises(ReplayError):
            read_bytes(tmp_path, data[:n])


def test_flipped_bytes_raise_replay_error_or_load(tmp_path: Path) -> None:
    w, s = make_run(burns=2)
    p = tmp_path / "a.swr"
    write(p, w, s, 1)
    data = p.read_bytes()
    rng = np.random.default_rng(1234)
    for pos in rng.integers(0, len(data), size=1000).tolist():
        bad = bytearray(data)
        bad[pos] ^= int(rng.integers(1, 256))
        try:
            read_bytes(tmp_path, bytes(bad))
        except ReplayError:
            pass


def test_wrong_magic(good: tuple[bytes, World, Scenario], tmp_path: Path) -> None:
    with pytest.raises(ReplayError, match="magic"):
        read_bytes(tmp_path, b"SWR1" + good[0][4:])


@pytest.mark.parametrize("version", [1, 3, "2", None, True])
def test_wrong_version(good: tuple[bytes, World, Scenario], tmp_path: Path, version: Any) -> None:
    h, body = split(good[0])
    h["version"] = version
    with pytest.raises(ReplayError, match="version"):
        read_bytes(tmp_path, join(h, body))


def test_header_length_beyond_the_file(good: tuple[bytes, World, Scenario], tmp_path: Path) -> None:
    data = good[0]
    bad = data[:4] + struct.pack("<I", len(data)) + data[8:]
    with pytest.raises(ReplayError, match="header"):
        read_bytes(tmp_path, bad)
    huge = data[:4] + struct.pack("<I", 0xFFFFFFFF) + data[8:]
    with pytest.raises(ReplayError):
        read_bytes(tmp_path, huge)


def test_invalid_utf8_and_json(tmp_path: Path) -> None:
    for raw in (b"\xff\xfe\xfd", b"{not json", b"[1,2]", b'{"a":NaN}', b"[" * 100_000):
        data = b"SWR2" + struct.pack("<I", len(raw)) + raw
        with pytest.raises(ReplayError):
            read_bytes(tmp_path, data)


HEADER_KEYS = [
    "version", "sim_version", "scenario", "scenario_hash", "seed", "swap", "horizon_us",
    "units", "rocks", "events", "sections",
]  # fmt: skip


@pytest.mark.parametrize("key", HEADER_KEYS)
def test_missing_header_field(
    good: tuple[bytes, World, Scenario], tmp_path: Path, key: str
) -> None:
    h, body = split(good[0])
    del h[key]
    with pytest.raises(ReplayError):
        read_bytes(tmp_path, join(h, body))


@pytest.mark.parametrize("key", [k for k in HEADER_KEYS if k != "version"])
@pytest.mark.parametrize("value", [None, "x", 1.5, [], {}, True])
def test_wrongly_typed_header_field(
    good: tuple[bytes, World, Scenario], tmp_path: Path, key: str, value: Any
) -> None:
    h, body = split(good[0])
    if (key, value) in [("sim_version", "x"), ("events", []), ("swap", True)]:
        pytest.skip("a legal value for that field")
    h[key] = value
    with pytest.raises(ReplayError):
        read_bytes(tmp_path, join(h, body))


def test_scenario_hash_must_match_the_scenario(
    good: tuple[bytes, World, Scenario], tmp_path: Path
) -> None:
    h, body = split(good[0])
    h["scenario"]["duration"] += 1.0
    with pytest.raises(ReplayError, match="hash"):
        read_bytes(tmp_path, join(h, body))


def test_bad_nested_fields(good: tuple[bytes, World, Scenario], tmp_path: Path) -> None:
    def case(mutate: Any) -> None:
        h, body = split(good[0])
        mutate(h)
        with pytest.raises(ReplayError):
            read_bytes(tmp_path, join(h, body))

    case(lambda h: h["units"][0].pop("birth"))
    case(lambda h: h["units"][1].update(id=7))
    case(lambda h: h["units"][0].update(end=-5))
    case(lambda h: h["units"][0].update(birth=3))
    case(lambda h: h["units"][0].update(team=4))
    case(lambda h: h["rocks"].update(radii=h["rocks"]["radii"][:-1]))
    case(lambda h: h["rocks"]["radii"].__setitem__(0, -1.0))
    case(
        lambda h: h["events"].append(
            {"t_us": 0, "type": "x", "unit": 999, "other": None, "data": {}}
        )
    )
    case(lambda h: h["events"].append({"t_us": 0, "type": "x", "unit": 0, "other": None}))
    case(
        lambda h: h["events"].append(
            {"t_us": 1.5, "type": "x", "unit": 0, "other": None, "data": {}}
        )
    )


def test_sections_must_match_the_bytes(good: tuple[bytes, World, Scenario], tmp_path: Path) -> None:
    def case(mutate: Any, body_edit: Any = lambda b: b) -> None:
        h, body = split(good[0])
        mutate(h)
        with pytest.raises(ReplayError):
            read_bytes(tmp_path, join(h, body_edit(body)))

    case(lambda h: h["sections"][0].update(count=h["sections"][0]["count"] + 1))
    case(lambda h: h["sections"][0].update(count=h["sections"][0]["count"] - 1))
    case(lambda h: h["sections"][0].update(count=-1))
    case(lambda h: h["sections"][0].update(count=2**62))
    case(lambda h: h["sections"][0].update(count=1.0))
    case(lambda h: h["sections"][0].update(dtype=[["unit", "<i8"]]))
    case(lambda h: h["sections"][0].update(dtype="x"))
    case(lambda h: h["sections"][0].update(dtype=[["unit", "<i4", [2**40]]]))
    case(lambda h: h["sections"][0].update(name="other"))  # segments missing
    case(lambda h: h["sections"].append(dict(h["sections"][0])))  # duplicate name
    case(lambda h: None, lambda b: b + b"\x00")  # trailing bytes
    case(lambda h: None, lambda b: b[:-1])
    case(
        lambda h: h["sections"].append({"name": "future", "dtype": [["x", "<f8"]], "count": 1,
                                        "required": True}),
        lambda b: b + b"\x00" * 8,
    )  # fmt: skip


def seg_edit(good: bytes, edit: Any) -> tuple[dict[str, Any], bytes]:
    h, body = split(good)
    n = h["sections"][0]["count"]
    dt = np.dtype(
        [("unit", "<i4"), ("t0", "<i8"), ("pos", "<f8", (3,)), ("vel", "<f8", (3,)),
         ("acc", "<f8", (3,))]
    )  # fmt: skip
    rows = np.frombuffer(body, dtype=dt, count=n).copy()
    edit(rows)
    return h, rows.tobytes()


def test_segments_must_be_ordered_known_and_finite(
    good: tuple[bytes, World, Scenario], tmp_path: Path
) -> None:
    def swap_rows(r: Any) -> None:
        r[[0, 1]] = r[[1, 0]]

    def case(edit: Any) -> None:
        h, body = seg_edit(good[0], edit)
        with pytest.raises(ReplayError):
            read_bytes(tmp_path, join(h, body))

    case(swap_rows)
    case(lambda r: r.__setitem__("unit", np.where(np.arange(len(r)) == 3, 999, r["unit"])))
    case(lambda r: r.__setitem__("unit", np.where(np.arange(len(r)) == 3, -1, r["unit"])))
    case(lambda r: r["pos"].__setitem__((2, 1), np.nan))
    case(lambda r: r["vel"].__setitem__((2, 1), np.inf))
    case(lambda r: r["acc"].__setitem__((2, 1), -np.inf))
    case(lambda r: r["t0"].__setitem__(1, 2**62))
    case(lambda r: r.__setitem__("t0", np.where(r["unit"] == 1, r["t0"] + 1, r["t0"])))  # birth


def test_unknown_extra_sections_are_skipped(
    good: tuple[bytes, World, Scenario], tmp_path: Path
) -> None:
    h, body = split(good[0])
    h["sections"].append(
        {"name": "beliefs", "dtype": [["a", "<i4"], ["b", "<f8", [2]]], "count": 3}
    )
    h["sections"].insert(0, {"name": "early", "dtype": [["a", "<u1"]], "count": 5})
    # Section bytes follow in listed order: the early section comes first.
    r = read_bytes(tmp_path, join(h, b"\x07" * 5 + body + b"\x01" * (3 * 20)))
    assert r.state(1, 0)[0].shape == (3,)
    assert [s["name"] for s in r.header["sections"]] == ["early", "segments", "beliefs"]


# ---- size ----


def test_a_24_hour_two_outpost_replay_is_under_5_mb(tmp_path: Path) -> None:
    s = load(EXAMPLE).model_copy(update={"duration": 86_400.0})
    w = build_world(s, 1)
    burn_all(w, 30)
    ships = [u for u in w.units if u[2] != "outpost"]
    assert len(ships) == 8
    p = tmp_path / "big.swr"
    write(p, w, s, 1)
    size = p.stat().st_size
    print(f"24 h two_outposts replay: {size} bytes, {len(w.events)} events")
    assert size < 5_000_000
    r = read(p)
    assert r.state(1, 43_200_000_000)[0].shape == (3,)
