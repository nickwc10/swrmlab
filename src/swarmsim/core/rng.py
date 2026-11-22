"""Keyed random draws.

Every draw is a pure function of (seed, stream, index). There is no shared generator
state, so drawing from one stream can never change another.
"""

from enum import IntEnum

import numpy as np
from numpy.typing import NDArray


class Stream(IntEnum):
    MAP = 0
    SPAWN = 1
    SENSING = 2
    ENGAGEMENT = 3
    COMMS = 4
    POLICY_BLUE = 5
    POLICY_RED = 6


def uniform(seed: int, stream: Stream, index: int, shape: tuple[int, ...]) -> NDArray[np.float64]:
    """Return float64 values in [0, 1) for the given seed, stream and index."""
    if seed < 0:
        raise ValueError(f"seed must be non-negative, got {seed}")
    if index < 0:
        raise ValueError(f"index must be non-negative, got {index}")
    bit_generator = np.random.Philox(key=[seed, int(stream)], counter=[0, 0, 0, index])
    return np.random.Generator(bit_generator).random(shape)
