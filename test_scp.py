"""
Tests for the Smart Coordinates Predictor (scp package).
"""

import pytest
from scp.models import Coordinate, CoordinateSequence
from scp.predictor import SmartCoordinatesPredictor


# ---------------------------------------------------------------------------
# Coordinate model tests
# ---------------------------------------------------------------------------

def test_coordinate_valid():
    c = Coordinate(latitude=37.7749, longitude=-122.4194)
    assert c.latitude == 37.7749
    assert c.longitude == -122.4194


def test_coordinate_invalid_latitude():
    with pytest.raises(ValueError):
        Coordinate(latitude=91.0, longitude=0.0)


def test_coordinate_invalid_longitude():
    with pytest.raises(ValueError):
        Coordinate(latitude=0.0, longitude=181.0)


def test_coordinate_as_tuple():
    c = Coordinate(latitude=10.0, longitude=20.0)
    assert c.as_tuple() == (10.0, 20.0)


# ---------------------------------------------------------------------------
# CoordinateSequence tests
# ---------------------------------------------------------------------------

def test_sequence_add_and_len():
    seq = CoordinateSequence()
    seq.add(Coordinate(0.0, 0.0)).add(Coordinate(1.0, 1.0))
    assert len(seq) == 2


def test_sequence_latest():
    seq = CoordinateSequence()
    assert seq.latest() is None
    seq.add(Coordinate(5.0, 5.0))
    assert seq.latest().latitude == 5.0


def test_sequence_empty():
    seq = CoordinateSequence()
    assert seq.is_empty()
    seq.add(Coordinate(0.0, 0.0))
    assert not seq.is_empty()


# ---------------------------------------------------------------------------
# Predictor tests — linear strategy
# ---------------------------------------------------------------------------

def _make_sequence(*lat_lon_pairs):
    seq = CoordinateSequence()
    for lat, lon in lat_lon_pairs:
        seq.add(Coordinate(latitude=lat, longitude=lon))
    return seq


def test_linear_single_step_straight_line():
    """Points moving +1 lat, +1 lon each step — next should be (3, 3)."""
    seq = _make_sequence((0.0, 0.0), (1.0, 1.0), (2.0, 2.0))
    predictor = SmartCoordinatesPredictor(strategy="linear", steps=1)
    preds = predictor.predict(seq)
    assert len(preds) == 1
    assert abs(preds[0].latitude - 3.0) < 1e-6
    assert abs(preds[0].longitude - 3.0) < 1e-6


def test_linear_multiple_steps():
    seq = _make_sequence((0.0, 0.0), (1.0, 1.0), (2.0, 2.0))
    predictor = SmartCoordinatesPredictor(strategy="linear", steps=3)
    preds = predictor.predict(seq)
    assert len(preds) == 3
    for i, pred in enumerate(preds, start=1):
        expected = 2.0 + i
        assert abs(pred.latitude - expected) < 1e-6


# ---------------------------------------------------------------------------
# Predictor tests — velocity strategy
# ---------------------------------------------------------------------------

def test_velocity_single_step():
    """delta=(1,1) between last two points — next should be (3, 3)."""
    seq = _make_sequence((0.0, 0.0), (1.0, 1.0), (2.0, 2.0))
    predictor = SmartCoordinatesPredictor(strategy="velocity", steps=1)
    preds = predictor.predict(seq)
    assert len(preds) == 1
    assert abs(preds[0].latitude - 3.0) < 1e-6
    assert abs(preds[0].longitude - 3.0) < 1e-6


def test_velocity_multiple_steps():
    seq = _make_sequence((10.0, 20.0), (10.5, 20.5))
    predictor = SmartCoordinatesPredictor(strategy="velocity", steps=4)
    preds = predictor.predict(seq)
    assert len(preds) == 4
    expected_lats = [11.0, 11.5, 12.0, 12.5]
    for pred, exp_lat in zip(preds, expected_lats):
        assert abs(pred.latitude - exp_lat) < 1e-6


# ---------------------------------------------------------------------------
# Predictor edge-case tests
# ---------------------------------------------------------------------------

def test_requires_at_least_two_points():
    seq = _make_sequence((0.0, 0.0))
    predictor = SmartCoordinatesPredictor()
    with pytest.raises(ValueError, match="At least 2"):
        predictor.predict(seq)


def test_invalid_strategy():
    with pytest.raises(ValueError, match="Unknown strategy"):
        SmartCoordinatesPredictor(strategy="magic")


def test_invalid_steps():
    with pytest.raises(ValueError, match="steps must be"):
        SmartCoordinatesPredictor(steps=0)


def test_latitude_clamped_at_boundary():
    """Prediction should not exceed ±90 latitude."""
    seq = _make_sequence((89.0, 0.0), (89.5, 0.0))
    predictor = SmartCoordinatesPredictor(strategy="velocity", steps=5)
    preds = predictor.predict(seq)
    for pred in preds:
        assert pred.latitude <= 90.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
