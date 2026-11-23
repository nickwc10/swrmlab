"""Keyed random draws.

Every draw is a pure function of (seed, stream, index). There is no shared generator
state, so drawing from one stream can never change another.
"""

import operator
from enum import IntEnum

import numpy as np
from numpy.typing import NDArray

# Seeds and indices must be below this, so every one of them is an exact 64-bit key word.
KEY_LIMIT = 2**63


class Stream(IntEnum):
    MAP = 0
    SPAWN = 1
    SENSING = 2
    ENGAGEMENT = 3
    COMMS = 4
    POLICY_BLUE = 5
    POLICY_RED = 6


def _key_word(name: str, value: int) -> int:
    try:
        # bool is an int subclass, but a truth value as a seed or index is a mistake.
        word = -1 if isinstance(value, bool) else operator.index(value)
    except TypeError:
        word = -1
    if not 0 <= word < KEY_LIMIT:
        raise ValueError(f"{name} must be an integer with 0 <= {name} < 2**63, got {value!r}")
    return word


def uniform(seed: int, stream: Stream, index: int, shape: tuple[int, ...]) -> NDArray[np.float64]:
    """Return float64 values in [0, 1) for the given seed, stream and index.

    ``seed`` and ``index`` must be integers from 0 to 2**63 - 1; anything else, including a
    bool, raises ValueError.
    """
    seed, index = _key_word("seed", seed), _key_word("index", index)
    # Explicit uint64 arrays: a list of large integers could be converted through float.
    key = np.array([seed, int(stream)], dtype=np.uint64)
    counter = np.array([0, 0, 0, index], dtype=np.uint64)
    return np.random.Generator(np.random.Philox(key=key, counter=counter)).random(shape)
