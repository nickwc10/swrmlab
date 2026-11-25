# swrmlab-sim

A deterministic simulator for autonomous space battles, with tools to analyze the tactics that
decide them.

Status: early development. Ships fly, burn fuel and hit rocks, light is delayed, and runs are
saved as replays. There is no sensing, no weapons and no tactics yet.

## About

- Independent open-source research.
- Findings are claims about the simulated world. They are not evidence about real systems.

## Setup

Requires [uv](https://docs.astral.sh/uv/). Python 3.12 is pinned in `.python-version`.

```
uv sync
uv run pytest
```

## Try it

Fly a scripted crossing: each side's ships cross 300,000 km to the other side's home cluster and
stop there. Then draw it.

```
uv run swarmsim demo --seed 1 --out data/replays/demo_seed1.swr
uv run swarmsim plot data/replays/demo_seed1.swr --out data/renders/demo_seed1.png
```

The same seed gives the same replay file, on one platform with the locked
dependencies. `--scenario FILE` picks another scenario; the default is
`scenarios/two_outposts.yaml` in the current folder. The demo does not steer around rocks between
the two home clusters, so on a few seeds a ship hits one.

The picture shows the paths from above, and the true distance from each outpost to one enemy ship
minus the distance that outpost sees, because light takes time to arrive.

Measure the speed of the core on your machine:

```
uv run python tests/bench/bench_core.py
```

## License

Apache-2.0. See LICENSE.
