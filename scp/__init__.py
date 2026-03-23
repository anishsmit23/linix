"""
SCP — Smart Coordinates Predictor

A lightweight library for predicting future GPS coordinates from
historical location data using linear extrapolation and velocity-based
forecasting.
"""

from .models import Coordinate, CoordinateSequence
from .predictor import SmartCoordinatesPredictor

__all__ = ["Coordinate", "CoordinateSequence", "SmartCoordinatesPredictor"]
__version__ = "0.1.0"
