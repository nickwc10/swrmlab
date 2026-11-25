# Contributing

Contributions are welcome. This is a small project run in spare time, so reviews may take a while.

## What to send

- **Bug fixes, tests, and documentation:** open a pull request directly.
- **New features, and any change under `src/swarmsim/core/`:** open an issue first to describe what
  you want to change and why. The engine is being reworked, and a quick discussion avoids work
  that conflicts with it.

## Setup

Requires [uv](https://docs.astral.sh/uv/). Python 3.12 is pinned in `.python-version`.

```
uv sync
```

## Checks

Every pull request must pass these, and CI runs them on Linux and macOS:

```
uv run pytest
uv run ruff check . && uv run ruff format --check .
uv run mypy src
```

Guidelines:

- Keep changes small and focused, with tests for new behavior.
- Runs must stay deterministic: the same seed gives the same replay, byte for byte. Never use
  wall-clock time, unseeded randomness, or iteration order that can vary between runs.
- Use SI units. Every setting needs a documented default, an allowed range, and a description (a
  test checks this).
- Use conventional commit messages, such as `feat:`, `fix:`, `test:`, `docs:`, `perf:`.

## Sign-off

Contributions are accepted under the project's Apache-2.0 license. Sign off every commit to certify
the [Developer Certificate of Origin](https://developercertificate.org/), which states that you
wrote the change or otherwise have the right to submit it:

```
git commit -s
```

By signing off you also confirm that the change uses only publicly available information. Do not
submit non-public, proprietary, classified, or export-controlled material.

## Reporting bugs

Open an issue with the version (`swarmsim.SIM_VERSION`), your Python version and platform, the
steps or code that reproduce the problem, and what you expected to happen.
