# swrmlab-sim

A deterministic simulator for autonomous space battles, with tools to analyze the tactics that
decide them.

Status: early development. Nothing is playable or usable yet; the repository currently holds
the project scaffold and a keyed random number module.

## About

- Independent open-source research.
- Built only from publicly available information.
- All units are fictional and generic.
- Findings are claims about the simulated world. They are not evidence about real systems.
- Contributions are not accepted. See CONTRIBUTING.md for how to report bugs.

## Setup

Requires [uv](https://docs.astral.sh/uv/). Python 3.12 is pinned in `.python-version`.

```
uv sync
uv run pytest
```

## License

Apache-2.0. See LICENSE.
