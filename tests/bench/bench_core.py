"""Speed of the core on this machine. Run: uv run python tests/bench/bench_core.py

Everything runs on one thread. A pair query is one emission_time plus one first_within,
through the public batch functions, so input checks are included. Each figure is the
median of several timed repeats; compile time is measured first and kept separate.
"""

import platform
import statistics
import subprocess
import sys
import time
from collections.abc import Callable
from pathlib import Path

import numba
import numpy as np
from numpy.typing import NDArray

from swarmsim import SIM_VERSION
from swarmsim.core.kernels import NOT_SEEN, emission_time_batch, first_within_batch, pack
from swarmsim.core.segments import SegmentStore
from swarmsim.demo import run_demo
from swarmsim.scenario import build_world, load

TARGET = 1_000_000
PAIRS = 2_000_000
UNITS = 100
HORIZON_US = 172_800_000_000
REPEATS = 5
SCENARIO = Path(__file__).parents[2] / "scenarios" / "two_outposts.yaml"


def median_time(f: Callable[[], object], repeats: int = REPEATS) -> float:
    times = []
    for _ in range(repeats):
        start = time.perf_counter()
        f()
        times.append(time.perf_counter() - start)
    return statistics.median(times)


def fleet(segments: int) -> SegmentStore:
    """Units flying 48 h starting in a 3e8 m box, with evenly spaced random burns up to 1 m/s2."""
    rng = np.random.default_rng(1)
    store = SegmentStore()
    for _ in range(UNITS):
        u = store.add_unit(0, rng.uniform(-1.5e8, 1.5e8, 3), rng.normal(0, 5e3, 3), (0, 0, 0))
        for k in range(1, segments):
            store.set_acc(u, HORIZON_US * k // segments, rng.uniform(-0.5, 0.5, 3))
    return store


def directions(rng: np.random.Generator, n: int) -> NDArray[np.float64]:
    v = rng.normal(size=(n, 3))
    return np.asarray(v / np.linalg.norm(v, axis=1)[:, None])


def contacts() -> tuple[NDArray[np.float64], ...]:
    """The mix the target was set on: ranges 3e8 to 3e9 m (ships) or 1e4 to 1e7 m (missiles,
    20%), half aimed to pass within two radii, accelerations up to 2 or 100 m/s2."""
    rng = np.random.default_rng(4)
    missile = rng.random(PAIRS) < 0.2
    far = np.where(missile, rng.uniform(1e4, 1e7, PAIRS), rng.uniform(3e8, 3e9, PAIRS))
    toward = directions(rng, PAIRS)
    radius = rng.uniform(10.0, 1e4, PAIRS)
    side = np.cross(toward, directions(rng, PAIRS))
    side /= np.linalg.norm(side, axis=1)[:, None]
    aim = toward + side * (rng.uniform(0.0, 2.0 * radius) / far)[:, None]
    aim /= np.linalg.norm(aim, axis=1)[:, None]
    aimed = (rng.random(PAIRS) < 0.5)[:, None]
    speed = rng.uniform(100.0, 1e5, PAIRS)[:, None]
    acc = (np.where(missile, 100.0, 2.0) * rng.uniform(0.0, 1.0, PAIRS))[:, None]
    dv = np.where(aimed, aim, directions(rng, PAIRS)) * speed
    aimed_acc = toward * (rng.random(PAIRS) < 0.7)[:, None]
    da = np.where(aimed, aimed_acc, directions(rng, PAIRS)) * acc
    return -toward * far[:, None], dv, da, radius, np.where(missile, 3600.0, 86_400.0)


def pair_queries(segments: int) -> tuple[float, float, float, float]:
    """(seconds for PAIRS emission times, for PAIRS contacts, share seen, share hit)."""
    rng = np.random.default_rng(2)
    paths = pack(fleet(segments))
    unit = rng.integers(0, UNITS, PAIRS)
    t_obs = rng.integers(0, HORIZON_US, PAIRS)
    early = rng.random(PAIRS) < 0.15  # soon after birth: light often has not arrived
    t_obs[early] = rng.integers(0, 12_000_000, int(early.sum()))
    pos = directions(rng, PAIRS) * rng.uniform(3e8, 3e9, PAIRS)[:, None]
    q = contacts()
    seen = float(np.mean(emission_time_batch(paths, unit, t_obs, pos) != NOT_SEEN))
    hit = float(np.mean(~np.isnan(first_within_batch(*q))))
    e = median_time(lambda: emission_time_batch(paths, unit, t_obs, pos))
    f = median_time(lambda: first_within_batch(*q))
    return e, f, seen, hit


def compile_time() -> float:
    """Seconds to compile both kernels, from the first (tiny) call of each."""
    store = fleet(1)
    start = time.perf_counter()
    emission_time_batch(pack(store), [0], [1], np.zeros((1, 3)))
    first_within_batch(np.ones((1, 3)), np.ones((1, 3)), np.ones((1, 3)), [0.1], [1.0])
    return time.perf_counter() - start


def engine_events() -> tuple[int, float]:
    """(events logged, median seconds): 8 ships, 500 random burn changes each, over 4 h."""
    scenario = load(SCENARIO)
    rng = np.random.default_rng(3)
    burns = [(int(t), a) for t, a in zip(
        np.sort(rng.integers(0, 14_400_000_000, 500)), rng.uniform(-0.1, 0.1, (500, 3)),
        strict=True,
    )]  # fmt: skip
    times, events = [], 0
    for _ in range(REPEATS):
        world = build_world(scenario, 1)
        for u in world.units:
            if u[2] != "outpost":
                for t, a in burns:
                    world.command_acc(u[0], t, a)
        start = time.perf_counter()
        world.run_until(world.horizon_us)
        times.append(time.perf_counter() - start)
        events = len(world.events)
    return events, statistics.median(times)


def cpu_name() -> str:
    """The processor's marketing name where macOS offers it."""
    if sys.platform == "darwin":
        query = ["sysctl", "-n", "machdep.cpu.brand_string"]
        return subprocess.run(query, capture_output=True, text=True).stdout.strip()
    return platform.processor() or platform.machine()


def main() -> None:
    print(f"machine   {platform.platform()}, {cpu_name()}")
    print(f"versions  swarmsim {SIM_VERSION}, python {platform.python_version()}, "
          f"numpy {np.__version__}, numba {numba.__version__}")  # fmt: skip
    print(f"method    {PAIRS:,} pairs per repeat, median of {REPEATS} repeats, one thread\n")
    print(f"compile time, both kernels     {compile_time():.2f} s\n")
    for segments in (1, 50):
        e, f, seen, hit = pair_queries(segments)
        rate = PAIRS / (e + f)
        verdict = "MET" if rate >= TARGET else "NOT MET"
        print(f"pair queries/s, {segments:2d} segment(s)   {int(rate):>12,}"
              f"   target {TARGET:,}: {verdict}")  # fmt: skip
        print(f"    emission_time only /s      {int(PAIRS / e):>12,}   ({seen:.1%} seen)")
        print(f"    first_within only /s       {int(PAIRS / f):>12,}   ({hit:.1%} hit)")
    scenario = load(SCENARIO)
    build = median_time(lambda: build_world(scenario, 1))
    whole = median_time(lambda: run_demo(scenario, 1))
    events = len(run_demo(scenario, 1).events)
    print(f"\ndemo battles/s, incl. build    {1 / whole:>12.1f}   "
          f"({whole * 1e3:.1f} ms each, build {build * 1e3:.1f} ms; {events} events)")  # fmt: skip
    n, seconds = engine_events()
    print(f"engine events/s                {int(n / seconds):>12,}   ({n} events, run_until only)")


if __name__ == "__main__":
    main()
