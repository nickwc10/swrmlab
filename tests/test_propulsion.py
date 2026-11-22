import math

import pytest

from swarmsim.core.propulsion import burn_time, fuel_after


def test_fuel_after_follows_the_rocket_equation() -> None:
    assert fuel_after(1000.0, 2.0, 500.0, 3000.0) == pytest.approx(
        1000.0 * math.exp(-1.0 / 3.0), rel=1e-12
    )


def test_coasting_burns_nothing() -> None:
    assert fuel_after(1000.0, 0.0, 1e6, 3000.0) == 1000.0
    assert fuel_after(1000.0, 5.0, 0.0, 3000.0) == 1000.0


def test_burn_time_gives_the_ideal_delta_v() -> None:
    acc, ve = 2.5, 3000.0
    t = burn_time(900.0, 300.0, acc, ve)
    assert acc * t == pytest.approx(ve * math.log(3.0), rel=1e-12)
    assert fuel_after(900.0, acc, t, ve) == pytest.approx(300.0, rel=1e-12)


def test_burn_time_edges() -> None:
    assert burn_time(300.0, 300.0, 1.0, 3000.0) == 0.0
    assert burn_time(900.0, 300.0, 0.0, 3000.0) == math.inf


def test_burn_time_keeps_precision_with_a_trace_of_fuel() -> None:
    assert burn_time(3.0 + 2.0**-40, 3.0, 1.0, 1000.0) == pytest.approx(
        1000.0 * 2.0**-40 / 3.0, rel=1e-9, abs=0.0
    )


@pytest.mark.parametrize(
    "args",
    [
        (0.0, 1.0, 1.0, 1.0),
        (1.0, -1.0, 1.0, 1.0),
        (1.0, 1.0, -1.0, 1.0),
        (1.0, 1.0, 1.0, 0.0),
        (math.nan, 1.0, 1.0, 1.0),
    ],
)
def test_fuel_after_rejects_bad_values(args: tuple[float, float, float, float]) -> None:
    with pytest.raises(ValueError):
        fuel_after(*args)


@pytest.mark.parametrize(
    "args",
    [
        (1.0, 2.0, 1.0, 1.0),  # less than dry mass
        (2.0, 0.0, 1.0, 1.0),
        (2.0, 1.0, -1.0, 1.0),
        (2.0, 1.0, 1.0, 0.0),
        (2.0, 1.0, math.inf, 1.0),
    ],
)
def test_burn_time_rejects_bad_values(args: tuple[float, float, float, float]) -> None:
    with pytest.raises(ValueError):
        burn_time(*args)
