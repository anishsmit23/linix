"""
Data models for the Smart Coordinates Predictor.
"""

from dataclasses import dataclass, field
from typing import List, Optional
from datetime import datetime


@dataclass
class Coordinate:
    """Represents a single GPS coordinate with an optional timestamp."""

    latitude: float
    longitude: float
    timestamp: Optional[datetime] = None

    def __post_init__(self):
        if not (-90 <= self.latitude <= 90):
            raise ValueError(f"Latitude must be between -90 and 90, got {self.latitude}")
        if not (-180 <= self.longitude <= 180):
            raise ValueError(f"Longitude must be between -180 and 180, got {self.longitude}")

    def as_tuple(self) -> tuple:
        """Return (latitude, longitude) as a plain tuple."""
        return (self.latitude, self.longitude)

    def __repr__(self) -> str:
        ts = f", ts={self.timestamp.isoformat()}" if self.timestamp else ""
        return f"Coordinate(lat={self.latitude:.6f}, lon={self.longitude:.6f}{ts})"


@dataclass
class CoordinateSequence:
    """An ordered sequence of Coordinate objects."""

    points: List[Coordinate] = field(default_factory=list)

    def add(self, coord: Coordinate) -> "CoordinateSequence":
        """Append a coordinate and return self for chaining."""
        self.points.append(coord)
        return self

    def __len__(self) -> int:
        return len(self.points)

    def __getitem__(self, index):
        return self.points[index]

    def __iter__(self):
        return iter(self.points)

    def is_empty(self) -> bool:
        return len(self.points) == 0

    def latest(self) -> Optional[Coordinate]:
        """Return the most recent coordinate, or None if empty."""
        return self.points[-1] if self.points else None
