"""
Core prediction logic for the Smart Coordinates Predictor.

Supported strategies
--------------------
* ``linear``  — least-squares linear extrapolation over the full history
* ``velocity``— uses only the last two points to derive an instant velocity
                and projects it forward by ``steps`` time units

Both strategies return a list of predicted :class:`~scp.models.Coordinate`
objects.
"""

from typing import List, Optional
from .models import Coordinate, CoordinateSequence


_STRATEGY_LINEAR = "linear"
_STRATEGY_VELOCITY = "velocity"
SUPPORTED_STRATEGIES = (_STRATEGY_LINEAR, _STRATEGY_VELOCITY)


class SmartCoordinatesPredictor:
    """Predict future GPS coordinates from a sequence of past observations.

    Parameters
    ----------
    strategy:
        Prediction algorithm — ``"linear"`` (default) or ``"velocity"``.
    steps:
        Number of future positions to predict (default: 1).
    """

    def __init__(self, strategy: str = _STRATEGY_LINEAR, steps: int = 1):
        if strategy not in SUPPORTED_STRATEGIES:
            raise ValueError(
                f"Unknown strategy '{strategy}'. Choose from {SUPPORTED_STRATEGIES}."
            )
        if steps < 1:
            raise ValueError("steps must be >= 1.")
        self.strategy = strategy
        self.steps = steps

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def predict(self, sequence: CoordinateSequence) -> List[Coordinate]:
        """Return a list of *steps* predicted coordinates.

        Parameters
        ----------
        sequence:
            Historical location data.  Must contain at least two points.

        Returns
        -------
        list[Coordinate]
            Predicted coordinates in chronological order.
        """
        if len(sequence) < 2:
            raise ValueError("At least 2 historical points are required for prediction.")

        if self.strategy == _STRATEGY_LINEAR:
            return self._predict_linear(sequence)
        return self._predict_velocity(sequence)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _predict_linear(self, sequence: CoordinateSequence) -> List[Coordinate]:
        """Least-squares linear extrapolation."""
        n = len(sequence)
        xs = list(range(n))  # time index

        lats = [c.latitude for c in sequence]
        lons = [c.longitude for c in sequence]

        lat_slope, lat_intercept = self._linear_fit(xs, lats)
        lon_slope, lon_intercept = self._linear_fit(xs, lons)

        predictions = []
        for step in range(1, self.steps + 1):
            t = n - 1 + step
            pred_lat = lat_slope * t + lat_intercept
            pred_lon = lon_slope * t + lon_intercept
            pred_lat = max(-90.0, min(90.0, pred_lat))
            pred_lon = max(-180.0, min(180.0, pred_lon))
            predictions.append(Coordinate(latitude=pred_lat, longitude=pred_lon))

        return predictions

    def _predict_velocity(self, sequence: CoordinateSequence) -> List[Coordinate]:
        """Velocity-based prediction using the last two points."""
        p1 = sequence[-2]
        p2 = sequence[-1]

        dlat = p2.latitude - p1.latitude
        dlon = p2.longitude - p1.longitude

        predictions = []
        for step in range(1, self.steps + 1):
            pred_lat = p2.latitude + dlat * step
            pred_lon = p2.longitude + dlon * step
            pred_lat = max(-90.0, min(90.0, pred_lat))
            pred_lon = max(-180.0, min(180.0, pred_lon))
            predictions.append(Coordinate(latitude=pred_lat, longitude=pred_lon))

        return predictions

    @staticmethod
    def _linear_fit(xs: list, ys: list):
        """Return (slope, intercept) for a simple least-squares fit."""
        n = len(xs)
        sum_x = sum(xs)
        sum_y = sum(ys)
        sum_xx = sum(x * x for x in xs)
        sum_xy = sum(x * y for x, y in zip(xs, ys))
        denom = n * sum_xx - sum_x ** 2
        if denom == 0:
            return 0.0, sum_y / n
        slope = (n * sum_xy - sum_x * sum_y) / denom
        intercept = (sum_y - slope * sum_x) / n
        return slope, intercept
