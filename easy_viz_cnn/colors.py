from itertools import cycle
from typing import NamedTuple


class RGB(NamedTuple):
    red: float
    green: float
    blue: float
    alpha: float = 1.0


DARK_LIGHT = cycle([RGB(0.4, 0.4, 0.4), RGB(0.8, 0.8, 0.8)])
BLUE_GRAY = cycle([RGB(0.654, 0.780, 0.905), RGB(0.8, 0.8, 0.8)])
GREEN_GRAY = cycle([RGB(0.5, 0.8, 0.5), RGB(0.8, 0.8, 0.8)])
ORANGE_BROWN = cycle([RGB(0.9, 0.6, 0.4), RGB(0.6, 0.4, 0.2)])
PURPLE_PINK = cycle([RGB(0.7, 0.5, 0.9), RGB(0.9, 0.7, 0.7)])
TEAL_GRAY = cycle([RGB(0.5, 0.8, 0.8), RGB(0.8, 0.8, 0.8)])
RED_BEIGE = cycle([RGB(0.9, 0.4, 0.4), RGB(0.9, 0.8, 0.7)])
SUNSET = cycle([
    RGB(0.9, 0.6, 0.3),  # Orange
    RGB(0.8, 0.4, 0.4),  # Red
    RGB(0.9, 0.7, 0.7),  # Pink
    RGB(0.7, 0.5, 0.9)   # Purple
])
OCEAN = cycle([
    RGB(0.2, 0.6, 0.8),  # Light Blue
    RGB(0.0, 0.5, 0.5),  # Teal
    RGB(0.4, 0.8, 0.6),  # Light Green
    RGB(0.2, 0.4, 0.7)   # Darker Blue
])
FOREST = cycle([
    RGB(0.3, 0.7, 0.3),  # Green
    RGB(0.4, 0.3, 0.2),  # Brown
    RGB(0.1, 0.4, 0.2),  # Dark Green
    RGB(0.1, 0.2, 0.4)   # Dark Blue
])
PASTEL = cycle([
    RGB(0.9, 0.7, 0.8),  # Pastel Pink
    RGB(0.7, 0.8, 0.9),  # Pastel Blue
    RGB(0.9, 0.9, 0.7),  # Pastel Yellow
    RGB(0.7, 0.9, 0.8)   # Pastel Green
])
RAINBOW = cycle([
    RGB(1.0, 0.0, 0.0),  # Red
    RGB(1.0, 0.5, 0.0),  # Orange
    RGB(1.0, 1.0, 0.0),  # Yellow
    RGB(0.0, 1.0, 0.0),  # Green
    RGB(0.0, 0.0, 1.0),  # Blue
    RGB(0.5, 0.0, 1.0)   # Purple
])
