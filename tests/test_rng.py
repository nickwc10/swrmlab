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


def test_negative_index_raises() -> None:
    with pytest.raises(ValueError):
        uniform(7, Stream.SPAWN, -1, (2,))


def test_negative_seed_raises() -> None:
    with pytest.raises(ValueError):
        uniform(-1, Stream.SPAWN, 0, (2,))
