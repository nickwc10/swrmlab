"""SWR2 replay files: a run's trajectory segments and events, seekable to any moment.

Layout, all little-endian:

* bytes 0-3: magic ``SWR2``
* bytes 4-7: uint32 header length ``H``
* bytes 8 to 8+H: the header, canonical UTF-8 JSON (sorted keys, compact separators, no NaN)
* then each section listed in ``header["sections"]``, in order, back to back
* a 32-byte trailer: the SHA-256 of every byte before it

The trailer detects damage (truncation, flipped bytes, hand edits). It is not a signature:
anyone can edit a file and recompute it, so it proves nothing about who wrote the file. The
header must also be exactly the canonical encoding of what it parses to, which rules out
duplicate keys and alternate spellings of the same values.

Each section entry is ``{name, dtype, count}`` (and optionally ``required: true``). ``dtype``
is a list of ``[field, type]`` or ``[field, type, shape]`` entries, so a reader can size any
section, including ones it does not know, and skip it. The one section defined today is
``segments``: the rows of ``SegmentStore.table()``. A later format adds sections without a
version change; a reader that meets an unknown section skips it unless it is marked required.

The header holds everything needed to identify and rebuild the run: the validated scenario and
its content hash, the seed, ``swap``, the horizon, units, rocks and events. The same run gives
the same bytes: nothing in the file depends on time, paths or library versions (only
``sim_version``).
"""

import hashlib
import json
import math
import operator
import os
import struct
import tempfile
from contextlib import suppress
from pathlib import Path
from typing import Any, NoReturn

import numpy as np
from numpy.typing import NDArray

from swarmsim import SIM_VERSION
from swarmsim.core.engine import Event, World
from swarmsim.core.geometry import Rocks
from swarmsim.core.rng import KEY_LIMIT
from swarmsim.core.segments import Vec3, _time, evaluate_segment
from swarmsim.scenario.build import build_world
from swarmsim.scenario.schema import Scenario, content_hash

MAGIC = b"SWR2"
VERSION = 2
_PREFIX = struct.Struct("<4sI")
_SEGMENTS = "segments"
_SEGMENT_SPEC: list[list[Any]] = [
    ["unit", "<i4"],
    ["t0", "<i8"],
    ["pos", "<f8", [3]],
    ["vel", "<f8", [3]],
    ["acc", "<f8", [3]],
]
_TYPES = frozenset(["<i1", "<u1", "<i2", "<u2", "<i4", "<u4", "<i8", "<u8", "<f4", "<f8"])
_MAX_FIELDS = 64
_MAX_SHAPE_ELEMENTS = 1 << 16
_T_LIMIT = 2**61  # same bound the segment store puts on times
_DIGEST = 32
_CLUSTER_LIMIT = 2**31


class ReplayError(ValueError):
    """The file is not a valid SWR2 replay."""


def _fail(message: str) -> NoReturn:
    raise ReplayError(message)


# ---- writing ----


def _dtype_from_spec(spec: list[list[Any]]) -> np.dtype[Any]:
    return np.dtype([(f[0], f[1], tuple(f[2])) if len(f) == 3 else (f[0], f[1]) for f in spec])


def _canonical(header: dict[str, Any]) -> bytes:
    return json.dumps(
        header, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
    ).encode("utf-8")


def _check_seed(seed: int) -> int:
    s = operator.index(seed)
    if not 0 <= s < KEY_LIMIT:
        raise ValueError(f"seed {s} must satisfy 0 <= seed < 2**63")
    return s


def _check_identity(world: World, scenario: Scenario, seed: int, swap: bool) -> None:
    """Raise ValueError unless ``world`` is what ``build_world(scenario, seed, swap)`` makes
    (plus any units added later), so the header cannot name a different run."""
    fresh = build_world(scenario, seed, swap)
    if world.horizon_us != fresh.horizon_us:
        raise ValueError(f"horizon_us is {world.horizon_us}, scenario gives {fresh.horizon_us}")
    if world.c != fresh.c:
        raise ValueError(f"c is {world.c}, scenario gives {fresh.c}")
    for name in ("centers", "radii", "cluster"):
        if not np.array_equal(getattr(world.rocks, name), getattr(fresh.rocks, name)):
            raise ValueError(f"rock {name} differ from what seed {seed}, swap={swap} build")
    if len(world.units) < len(fresh.units):
        raise ValueError(
            f"world has {len(world.units)} units, fewer than the {len(fresh.units)} built"
        )
    for want, have in zip(fresh.units, world.units, strict=False):
        unit = want[0]
        if have[:4] != want[:4]:
            raise ValueError(
                f"unit {unit} differs in team, class or birth from what seed/swap build"
            )
        a, b = fresh.store.state(unit, want[3]), world.store.state(unit, want[3])
        if not (np.array_equal(a[0], b[0]) and np.array_equal(a[1], b[1])):
            raise ValueError(
                f"unit {unit} starts at another position or velocity than seed/swap build"
            )


def write(
    path: str | Path, world: World, scenario: Scenario, seed: int, swap: bool = False
) -> None:
    """Write ``world``'s run to ``path``, atomically.

    ``scenario``, ``seed`` and ``swap`` must be what built the world; they are stored so the
    starting world can be rebuilt with ``build_world``.
    """
    seed = _check_seed(seed)
    _check_identity(world, scenario, seed, swap)
    table = world.store.table()
    seg_dtype = _dtype_from_spec(_SEGMENT_SPEC)
    if table.dtype != seg_dtype:
        raise ValueError(f"segment table has dtype {table.dtype}, expected {seg_dtype}")
    rocks = world.rocks
    header: dict[str, Any] = {
        "version": VERSION,
        "sim_version": SIM_VERSION,
        "scenario": scenario.model_dump(mode="json"),
        "scenario_hash": content_hash(scenario),
        "seed": seed,
        "swap": bool(swap),
        "horizon_us": world.horizon_us,
        "units": [
            {"id": i, "team": team, "cls": cls, "birth": birth, "end": end}
            for i, team, cls, birth, end in world.units
        ],
        "rocks": {
            "centers": rocks.centers.tolist(),
            "radii": rocks.radii.tolist(),
            "cluster": rocks.cluster.tolist(),
        },
        "events": [
            {"t_us": e.t_us, "type": e.type, "unit": e.unit, "other": e.other, "data": e.data}
            for e in world.events
        ],
        "sections": [{"name": _SEGMENTS, "dtype": _SEGMENT_SPEC, "count": len(table)}],
    }
    raw = _canonical(header)
    payload = _PREFIX.pack(MAGIC, len(raw)) + raw + table.tobytes()
    blob = payload + hashlib.sha256(payload).digest()

    target = Path(path)
    fd, tmp = tempfile.mkstemp(dir=target.parent, prefix=f".{target.name}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(blob)
            f.flush()
            os.fsync(f.fileno())
        os.chmod(tmp, 0o644)
        os.replace(tmp, target)
    except BaseException:
        with suppress(OSError):
            os.unlink(tmp)
        raise


# ---- reading ----


def _int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        _fail(f"{name} must be an integer")
    return value


def _str(value: Any, name: str) -> str:
    if not isinstance(value, str):
        _fail(f"{name} must be a string")
    return value


def _dict(value: Any, name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        _fail(f"{name} must be an object")
    return value


def _list(value: Any, name: str) -> list[Any]:
    if not isinstance(value, list):
        _fail(f"{name} must be a list")
    return value


def _number(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        _fail(f"{name} must be a number")
    return float(value)


def _field(header: dict[str, Any], key: str) -> Any:
    if key not in header:
        _fail(f"header is missing '{key}'")
    return header[key]


def _reject_constant(name: str) -> NoReturn:
    raise ValueError(f"{name} is not allowed in JSON")


def _finite_float(text: str) -> float:
    value = float(text)
    if not math.isfinite(value):
        raise ValueError(f"number {text[:20]} is not finite")
    return value


def _section_dtype(spec: Any, where: str) -> np.dtype[Any]:
    fields = _list(spec, f"{where}.dtype")
    if not 1 <= len(fields) <= _MAX_FIELDS:
        _fail(f"{where}.dtype must list 1 to {_MAX_FIELDS} fields")
    names: set[str] = set()
    parsed: list[tuple[Any, ...]] = []
    for f in fields:
        entry = _list(f, f"{where}.dtype entry")
        if len(entry) not in (2, 3):
            _fail(f"{where}.dtype entries need a name, a type and an optional shape")
        name, kind = _str(entry[0], f"{where}.dtype name"), _str(entry[1], f"{where}.dtype type")
        if not name or name in names:
            _fail(f"{where}.dtype has an empty or repeated field name")
        if kind not in _TYPES:
            _fail(f"{where}.dtype type '{kind}' is not supported")
        names.add(name)
        if len(entry) == 3:
            shape = _list(entry[2], f"{where}.dtype shape")
            dims = [_int(d, f"{where}.dtype shape") for d in shape]
            if not 1 <= len(dims) <= 3 or min(dims) < 1 or int(np.prod(dims)) > _MAX_SHAPE_ELEMENTS:
                _fail(f"{where}.dtype shape is out of range")
            parsed.append((name, kind, tuple(dims)))
        else:
            parsed.append((name, kind))
    return np.dtype(parsed)


def _parse_units(raw: Any, horizon: int) -> list[dict[str, Any]]:
    units = _list(raw, "units")
    for i, u in enumerate(units):
        u = _dict(u, "unit")
        if _int(_field(u, "id"), "unit id") != i:
            _fail("unit ids must be 0, 1, 2 ... in order")
        _str(_field(u, "team"), "unit team")
        _str(_field(u, "cls"), "unit cls")
        birth = _int(_field(u, "birth"), "unit birth")
        if not 0 <= birth <= horizon:
            _fail("unit birth must lie within the run")
        end = _field(u, "end")
        if end is not None and not birth <= _int(end, "unit end") <= horizon:
            _fail("unit end must lie between its birth and the horizon")
    return units


def _parse_events(raw: Any, n_units: int, horizon: int) -> tuple[Event, ...]:
    out: list[Event] = []
    for e in _list(raw, "events"):
        e = _dict(e, "event")
        t, unit = _int(_field(e, "t_us"), "event t_us"), _int(_field(e, "unit"), "event unit")
        if not 0 <= t <= horizon or (out and t < out[-1].t_us):
            _fail("event times must lie within the run and not decrease")
        other = _field(e, "other")
        if other is not None:
            other = _int(other, "event other")
        if not 0 <= unit < n_units or (other is not None and not 0 <= other < n_units):
            _fail("event names an unknown unit")
        etype, data = _str(_field(e, "type"), "event type"), _dict(_field(e, "data"), "data")
        out.append(Event(t, etype, unit, other, data))
    return tuple(out)


def _parse_rocks(raw: Any) -> Rocks:
    d = _dict(raw, "rocks")
    centers = [
        [_number(x, "rock center") for x in _list(c, "rock center")]
        for c in _list(_field(d, "centers"), "rocks.centers")
    ]
    radii = [_number(x, "rock radius") for x in _list(_field(d, "radii"), "rocks.radii")]
    cluster = [_int(x, "rock cluster") for x in _list(_field(d, "cluster"), "rocks.cluster")]
    if any(not 0 <= x < _CLUSTER_LIMIT for x in cluster):
        _fail("rock cluster ids are out of range")
    if any(len(c) != 3 for c in centers):
        _fail("rock centers need three components")
    return Rocks(np.array(centers).reshape(-1, 3), np.array(radii), np.array(cluster))


def _check_segments(
    rows: NDArray[np.void], units: list[dict[str, Any]], horizon: int
) -> tuple[NDArray[np.int64], NDArray[np.int64]]:
    """Validate segment rows; return the per-unit start offsets and the t0 column."""
    n = len(units)
    unit, t0 = rows["unit"].astype(np.int64), rows["t0"].astype(np.int64)
    if len(rows) and (unit.min() < 0 or unit.max() >= n):
        _fail("a segment names an unknown unit")
    if len(rows) and (t0.min() < 0 or t0.max() > horizon):
        _fail("a segment starts outside the run")
    for name in ("pos", "vel", "acc"):
        if not np.all(np.isfinite(rows[name])):
            _fail(f"segment {name} must be finite")
    if len(rows) > 1:
        same = unit[1:] == unit[:-1]
        if np.any(unit[1:] < unit[:-1]) or np.any(same & (t0[1:] <= t0[:-1])):
            _fail("segments must be ordered by unit, then by strictly increasing start time")
    starts = np.searchsorted(unit, np.arange(n + 1)).astype(np.int64)
    for i, u in enumerate(units):
        a, b = int(starts[i]), int(starts[i + 1])
        if a == b:
            _fail(f"unit {i} has no segments")
        if int(t0[a]) != u["birth"]:
            _fail(f"unit {i}'s first segment does not start at its birth")
        if u["end"] is not None and int(t0[b - 1]) > u["end"]:
            _fail(f"unit {i} has a segment after its end")
    return starts, t0


class Replay:
    """A loaded replay. Build one with ``read``."""

    def __init__(
        self,
        header: dict[str, Any],
        scenario: Scenario,
        rocks: Rocks,
        events: tuple[Event, ...],
        rows: NDArray[np.void],
        starts: NDArray[np.int64],
        t0: NDArray[np.int64],
    ) -> None:
        self.header = header
        self.scenario = scenario
        self.rocks = rocks
        self.events = events
        self._rows = rows
        self._starts = starts
        self._t0 = t0
        self._units = header["units"]

    def state(self, unit: int, t_us: int) -> tuple[Vec3, Vec3]:
        """Position and velocity of ``unit`` at ``t_us``, as ``SegmentStore.state`` gives.

        A unit that has not ended extrapolates its last segment.
        """
        u = operator.index(unit)
        if not 0 <= u < len(self._units):
            raise ValueError(f"unknown unit {unit}")
        t = _time("t_us", t_us)
        birth, end = self._units[u]["birth"], self._units[u]["end"]
        if t < birth:
            raise ValueError(f"t_us {t} is before unit {u} was born at {birth}")
        if end is not None and t > end:
            raise ValueError(f"t_us {t} is after unit {u} ended at {end}")
        a, b = int(self._starts[u]), int(self._starts[u + 1])
        row = self._rows[a + int(np.searchsorted(self._t0[a:b], t, side="right")) - 1]
        return evaluate_segment(int(row["t0"]), row["pos"], row["vel"], row["acc"], t)


def _parse(file: bytes) -> Replay:
    if len(file) < _PREFIX.size + _DIGEST:
        _fail("file is shorter than the fixed prefix and checksum")
    if file[: len(MAGIC)] != MAGIC:
        _fail(f"wrong magic {file[: len(MAGIC)]!r}, expected {MAGIC!r}")
    data, digest = file[:-_DIGEST], file[-_DIGEST:]
    if hashlib.sha256(data).digest() != digest:
        _fail("checksum mismatch: the file is damaged or was edited")
    _, header_len = _PREFIX.unpack_from(data)
    body = _PREFIX.size + header_len
    if body > len(data):
        _fail("header length runs past the end of the file")
    raw = data[_PREFIX.size : body]
    header = json.loads(
        raw.decode("utf-8"), parse_constant=_reject_constant, parse_float=_finite_float
    )
    header = _dict(header, "header")
    if _canonical(header) != raw:
        _fail("header is not in canonical form (duplicate keys, spacing or number spelling)")
    if _int(header.get("version"), "version") != VERSION:
        _fail(f"unsupported version {header.get('version')}, expected {VERSION}")
    _str(_field(header, "sim_version"), "sim_version")
    if not 0 <= _int(_field(header, "seed"), "seed") < KEY_LIMIT:
        _fail("seed is out of range")
    if not isinstance(_field(header, "swap"), bool):
        _fail("swap must be a boolean")
    horizon = _int(_field(header, "horizon_us"), "horizon_us")
    scenario = Scenario.model_validate(_dict(_field(header, "scenario"), "scenario"))
    if _canonical(header["scenario"]) != _canonical(scenario.model_dump(mode="json")):
        _fail("scenario is not in its canonical form")
    if _str(_field(header, "scenario_hash"), "scenario_hash") != content_hash(scenario):
        _fail("scenario_hash does not match the stored scenario")
    if horizon != round(scenario.duration * 1e6):
        _fail("horizon_us does not match the scenario duration")
    units = _parse_units(_field(header, "units"), horizon)
    rocks = _parse_rocks(_field(header, "rocks"))
    events = _parse_events(_field(header, "events"), len(units), horizon)

    offset = body
    rows: NDArray[np.void] | None = None
    seen: set[str] = set()
    for i, sec in enumerate(_list(_field(header, "sections"), "sections")):
        sec = _dict(sec, f"sections[{i}]")
        name = _str(sec.get("name"), "section name")
        if name in seen:
            _fail(f"section '{name}' appears twice")
        seen.add(name)
        dtype = _section_dtype(sec.get("dtype"), f"section '{name}'")
        count = _int(sec.get("count"), f"section '{name}' count")
        # Compare against the bytes that are really there before allocating anything.
        if count < 0 or count > (len(data) - offset) // dtype.itemsize:
            _fail(f"section '{name}' count does not fit in the file")
        end = offset + count * dtype.itemsize
        if name == _SEGMENTS:
            if sec["dtype"] != _SEGMENT_SPEC:
                _fail("segments section has the wrong dtype")
            rows = np.frombuffer(data, dtype=dtype, count=count, offset=offset)
        elif sec.get("required") is True:
            _fail(f"section '{name}' is required but unknown")
        offset = end
    if offset != len(data):
        _fail("trailing bytes after the last section")
    if rows is None:
        _fail("segments section is missing")
    starts, t0 = _check_segments(rows, units, horizon)
    return Replay(header, scenario, rocks, events, rows, starts, t0)


_BAD_CONTENT = (ValueError, TypeError, KeyError, IndexError, OverflowError, RecursionError)


def read(path: str | Path) -> Replay:
    """Load a replay. Any malformed content raises ``ReplayError``."""
    data = Path(path).read_bytes()
    try:
        return _parse(data)
    except ReplayError:
        raise
    except _BAD_CONTENT as err:
        raise ReplayError(f"malformed replay: {err}") from None
