import numpy as np
import pytest

from swarmsim.core.segments import SegmentStore

HOUR_US = 3_600_000_000


def test_constant_acceleration_covers_light_second_distance() -> None:
    store = SegmentStore()
    u = store.add_unit(0, (0, 0, 0), (0, 0, 0), (1, 0, 0))
    pos, vel = store.state(u, 24_494_897_428)
    assert abs(pos[0] - 3e8) < 1.0
    assert abs(vel[0] - 24_494.897428) < 1e-6


def test_accelerate_then_brake_covers_distance_and_stops() -> None:
    store = SegmentStore()
    u = store.add_unit(0, (0, 0, 0), (0, 0, 0), (1, 0, 0))
    half = 17_320_508_076
    store.set_acc(u, half, (-1, 0, 0))
    pos, vel = store.state(u, 2 * half)
    assert abs(pos[0] - 3e8) < 1.0
    assert abs(vel[0]) < 1e-3


def test_one_step_equals_many_set_acc_calls() -> None:
    day_us = 24 * HOUR_US
    one = SegmentStore()
    a = one.add_unit(0, (1, 2, 3), (4, 5, 6), (0.5, -0.25, 0.125))
    many = SegmentStore()
    b = many.add_unit(0, (1, 2, 3), (4, 5, 6), (0.5, -0.25, 0.125))
    for k in range(1, 1001):
        many.set_acc(b, k * day_us // 1000, (0.5, -0.25, 0.125))
    p1, v1 = one.state(a, day_us)
    p2, v2 = many.state(b, day_us)
    assert np.max(np.abs(p1 - p2)) < 1.0
    assert np.max(np.abs(v1 - v2)) < 1e-6


def test_position_and_velocity_continuous_across_change() -> None:
    store = SegmentStore()
    u = store.add_unit(0, (0, 0, 0), (1, 0, 0), (2, 0, 0))
    t = 5_000_000
    before_pos, before_vel = store.state(u, t)
    store.set_acc(u, t, (-3, 1, 0))
    after_pos, after_vel = store.state(u, t)
    assert np.array_equal(before_pos, after_pos) or np.allclose(before_pos, after_pos, atol=1e-12)
    assert np.allclose(before_vel, after_vel, atol=1e-12)
    # The earlier segment is untouched.
    pos, _ = store.state(u, 2_000_000)
    assert np.allclose(pos, (1 * 2 + 0.5 * 2 * 4, 0, 0))


def test_set_acc_before_last_segment_raises() -> None:
    store = SegmentStore()
    u = store.add_unit(0, (0, 0, 0), (0, 0, 0), (0, 0, 0))
    store.set_acc(u, 2_000_000, (1, 0, 0))
    with pytest.raises(ValueError, match="before"):
        store.set_acc(u, 1_000_000, (1, 0, 0))


def test_set_acc_at_same_instant_replaces_segment() -> None:
    store = SegmentStore()
    u = store.add_unit(0, (0, 0, 0), (0, 0, 0), (1, 0, 0))
    store.set_acc(u, 0, (2, 0, 0))
    store.set_acc(u, 1_000_000, (3, 0, 0))
    store.set_acc(u, 1_000_000, (4, 0, 0))
    table = store.table()
    assert len(table) == 2
    assert list(table["t0"]) == [0, 1_000_000]
    assert table["acc"][0][0] == 2.0
    assert table["acc"][1][0] == 4.0


def test_state_outside_lifetime_raises() -> None:
    store = SegmentStore()
    u = store.add_unit(1_000_000, (0, 0, 0), (0, 0, 0), (0, 0, 0))
    with pytest.raises(ValueError, match="before"):
        store.state(u, 999_999)
    store.end(u, 3_000_000)
    store.state(u, 3_000_000)
    with pytest.raises(ValueError, match="after"):
        store.state(u, 3_000_001)
    assert store.birth(u) == 1_000_000
    assert store.end_time(u) == 3_000_000


def test_unended_unit_extrapolates_and_end_time_is_none() -> None:
    store = SegmentStore()
    u = store.add_unit(0, (0, 0, 0), (1, 0, 0), (0, 0, 0))
    pos, _ = store.state(u, 10_000_000)
    assert pos[0] == 10.0
    assert store.end_time(u) is None


def test_invalid_operations_raise() -> None:
    store = SegmentStore()
    u = store.add_unit(1_000_000, (0, 0, 0), (0, 0, 0), (0, 0, 0))
    with pytest.raises(ValueError, match="unknown"):
        store.state(5, 2_000_000)
    with pytest.raises(ValueError, match="unknown"):
        store.birth(-1)
    with pytest.raises(ValueError, match="end"):
        store.end(u, 500_000)
    store.end(u, 2_000_000)
    with pytest.raises(ValueError, match="already ended"):
        store.end(u, 2_500_000)
    with pytest.raises(ValueError, match="ended"):
        store.set_acc(u, 2_500_000, (1, 0, 0))
    with pytest.raises(ValueError, match="finite"):
        store.add_unit(0, (float("nan"), 0, 0), (0, 0, 0), (0, 0, 0))
    with pytest.raises(ValueError, match="finite"):
        store.add_unit(0, (0, 0, 0), (float("inf"), 0, 0), (0, 0, 0))
    with pytest.raises(ValueError, match="length 3"):
        store.add_unit(0, (0, 0), (0, 0, 0), (0, 0, 0))


def test_inputs_are_copied_and_outputs_are_fresh() -> None:
    store = SegmentStore()
    pos_in = np.array([1.0, 2.0, 3.0])
    u = store.add_unit(0, pos_in, (0, 0, 0), (0, 0, 0))
    pos_in[0] = 99.0
    p, _ = store.state(u, 0)
    assert p[0] == 1.0
    p[1] = -7.0
    p2, _ = store.state(u, 0)
    assert p2[1] == 2.0
    t1 = store.table()
    t1["pos"][0][0] = 5.0
    assert store.table()["pos"][0][0] == 1.0


def test_table_order_dtype_and_growth() -> None:
    store = SegmentStore()
    ids = [store.add_unit(k, (k, 0, 0), (0, 0, 0), (0, 0, 0)) for k in range(100)]
    for k, u in enumerate(ids):
        store.set_acc(u, 1000 + k, (1, 0, 0))
    table = store.table()
    assert len(table) == 200
    assert table.dtype.names == ("unit", "t0", "pos", "vel", "acc")
    assert table.dtype["unit"] == np.dtype("<i4")
    assert table.dtype["t0"] == np.dtype("<i8")
    assert table.dtype["pos"] == np.dtype(("<f8", (3,)))
    assert list(table["unit"][:4]) == [0, 0, 1, 1]
    assert list(table["unit"]) == sorted(table["unit"])
    pos, _ = store.state(ids[50], 1050)
    assert pos[0] == 50.0
