from typing import Any

import numpy as np
import pytest

from swarmsim.core.rng import Stream, uniform


def test_stream_values() -> None:
    assert [int(s) for s in Stream] == [0, 1, 2, 3, 4, 5, 6]
    assert Stream.MAP == 0 and Stream.POLICY_RED == 6


def test_same_arguments_equal() -> None:
    a = uniform(7, Stream.SPAWN, 3, (4, 3))
    b = uniform(7, Stream.SPAWN, 3, (4, 3))
    assert a.shape == (4, 3)
    assert a.dtype == np.float64
    assert np.array_equal(a, b)


def test_range_half_open() -> None:
    x = uniform(1, Stream.MAP, 0, (10000,))
    assert x.min() >= 0.0 and x.max() < 1.0


@pytest.mark.parametrize(
    "other",
    [(8, Stream.SPAWN, 3), (7, Stream.SENSING, 3), (7, Stream.SPAWN, 4)],
)
def test_changing_any_key_changes_draw(other: tuple[int, Stream, int]) -> None:
    base = uniform(7, Stream.SPAWN, 3, (16,))
    assert not np.array_equal(base, uniform(*other, (16,)))


def test_other_streams_do_not_interfere() -> None:
    first = uniform(7, Stream.SPAWN, 3, (16,))
    for s in Stream:
        uniform(7, s, 3, (100,))
    uniform(7, Stream.ENGAGEMENT, 0, (5,))
    assert np.array_equal(first, uniform(7, Stream.SPAWN, 3, (16,)))


BAD_KEYS = [-1, 2**63, 2**64, 10**30, True, False, np.True_, 1.0, 1.5, float("nan"), "1", None]


@pytest.mark.parametrize("bad", BAD_KEYS)
def test_bad_seed_raises_and_is_named(bad: Any) -> None:
    with pytest.raises(ValueError, match="seed"):
        uniform(bad, Stream.SPAWN, 0, (2,))


@pytest.mark.parametrize("bad", BAD_KEYS)
def test_bad_index_raises_and_is_named(bad: Any) -> None:
    with pytest.raises(ValueError, match="index"):
        uniform(7, Stream.SPAWN, bad, (2,))


def test_numpy_integers_give_the_same_draws_as_ints() -> None:
    a = uniform(np.int64(7), Stream.SPAWN, np.uint32(3), (4,))  # type: ignore[arg-type]
    assert np.array_equal(a, uniform(7, Stream.SPAWN, 3, (4,)))


# Values recorded before seeds and indices were bounded; they must never change.
GOLDEN = [
    (0, Stream.MAP, 0, [0.011546754286331562, 0.24154919656271812, 0.11142585551493822]),
    (1, Stream.SPAWN, 10_001_000, [0.1599636220906705, 0.6835539228362972, 0.8835505849101543]),
    (2**53, Stream.SENSING, 3, [0.7617025320628338, 0.2952386449283535, 0.7237019037517992]),
    (2**53 + 1, Stream.SENSING, 3, [0.5771428576198664, 0.1431261963200645, 0.3437613972021716]),
    (2**63 - 1, Stream.ENGAGEMENT, 0, [0.9138379561395334, 0.5548339494361428, 0.382087499939355]),
    (7, Stream.COMMS, 2**53, [0.9139040259861555, 0.40160527260342904, 0.401820010303935]),
    (7, Stream.COMMS, 2**53 + 1, [0.38054548354650264, 0.10131697183951227, 0.5943271904918087]),
    (
        7, Stream.POLICY_BLUE, 2**62 + 1,
        [0.8411455740208199, 0.19510514727467732, 0.5851064211733514],
    ),
    (
        2**63 - 1, Stream.POLICY_RED, 2**63 - 1,
        [0.3217404578241978, 0.9898759169770508, 0.39379849671088984],
    ),
]  # fmt: skip


@pytest.mark.parametrize(("seed", "stream", "index", "want"), GOLDEN)
def test_golden_values(seed: int, stream: Stream, index: int, want: list[float]) -> None:
    assert uniform(seed, stream, index, (3,)).tolist() == want


def test_neighbouring_large_seeds_and_indices_differ() -> None:
    a = uniform(2**62, Stream.SPAWN, 0, (16,))
    assert not np.array_equal(a, uniform(2**62 + 1, Stream.SPAWN, 0, (16,)))
    b = uniform(0, Stream.SPAWN, 2**62, (16,))
    assert not np.array_equal(b, uniform(0, Stream.SPAWN, 2**62 + 1, (16,)))


@pytest.mark.parametrize("index", [0, 2**53, 2**63 - 2])
def test_a_large_draw_shares_no_values_with_the_next_index(index: int) -> None:
    a = uniform(7, Stream.SPAWN, index, (10_000,))
    b = uniform(7, Stream.SPAWN, index + 1, (10_000,))
    assert np.intersect1d(a, b).size == 0
