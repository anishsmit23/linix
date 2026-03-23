"""
Command-line interface for the Smart Coordinates Predictor.

Usage
-----
    python -m scp.cli --help

Examples
--------
    # Predict next coordinate from three points (linear strategy)
    python -m scp.cli 37.7749,-122.4194 37.7800,-122.4150 37.7850,-122.4100

    # Use velocity strategy and predict 3 steps ahead
    python -m scp.cli --strategy velocity --steps 3 \
        48.8566,2.3522 48.8600,2.3600 48.8640,2.3680
"""

import argparse
import sys

from .models import Coordinate, CoordinateSequence
from .predictor import SmartCoordinatesPredictor, SUPPORTED_STRATEGIES


def parse_coord(value: str) -> Coordinate:
    """Parse a 'lat,lon' string into a Coordinate."""
    try:
        lat_str, lon_str = value.split(",")
        return Coordinate(latitude=float(lat_str), longitude=float(lon_str))
    except (ValueError, TypeError):
        raise argparse.ArgumentTypeError(
            f"Invalid coordinate '{value}'. Expected format: lat,lon  (e.g. 37.77,-122.41)"
        )


def main(argv=None):
    parser = argparse.ArgumentParser(
        prog="scp",
        description="SCP — Smart Coordinates Predictor: predict future GPS locations.",
    )
    parser.add_argument(
        "points",
        metavar="lat,lon",
        type=parse_coord,
        nargs="+",
        help="Historical GPS coordinates in chronological order (at least 2 required).",
    )
    parser.add_argument(
        "--strategy",
        choices=SUPPORTED_STRATEGIES,
        default="linear",
        help="Prediction strategy (default: linear).",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=1,
        metavar="N",
        help="Number of future positions to predict (default: 1).",
    )

    args = parser.parse_args(argv)

    if len(args.points) < 2:
        parser.error("At least 2 historical coordinate points are required.")

    sequence = CoordinateSequence(points=args.points)
    predictor = SmartCoordinatesPredictor(strategy=args.strategy, steps=args.steps)

    try:
        predictions = predictor.predict(sequence)
    except ValueError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)

    print(f"Strategy : {args.strategy}")
    print(f"History  : {len(sequence)} point(s)")
    print(f"Predicted: {len(predictions)} point(s)")
    print()
    for i, coord in enumerate(predictions, start=1):
        print(f"  Step {i}: lat={coord.latitude:.6f}, lon={coord.longitude:.6f}")


if __name__ == "__main__":
    main()
