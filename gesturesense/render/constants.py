"""Geometry and lighting constants shared by the renderer modules."""

from __future__ import annotations

import cv2
import numpy as np

VIEW_W, VIEW_H = 720, 540  # size of the image shown in the window
SCALE = 11.5  # metres -> scene units (a hand is ~1.6 units long)
HAND_HOME_Y = 1.55  # height of the hand above the floor
GAUGE_POS = (2.75, 0.0, 0.0)  # base centre of the volume tower
GAUGE_H, GAUGE_R, GAUGE_SLABS, GAUGE_FACETS = 2.7, 0.30, 20, 20
GRID_HALF, GRID_STEP = 5.0, 0.5
GRID_CENTER = (1.1, 0.0)  # (x, z) centre of the floor grid
UP = np.array([0.0, 1.0, 0.0])
FONT = cv2.FONT_HERSHEY_SIMPLEX

CONNECTIONS = [
    (0, 1),
    (1, 2),
    (2, 3),
    (3, 4),
    (0, 5),
    (5, 6),
    (6, 7),
    (7, 8),
    (5, 9),
    (9, 10),
    (10, 11),
    (11, 12),
    (9, 13),
    (13, 14),
    (14, 15),
    (15, 16),
    (13, 17),
    (17, 18),
    (18, 19),
    (19, 20),
    (0, 17),
]
PALM_EDGES = {(0, 5), (5, 9), (9, 13), (13, 17), (0, 17)}  # drawn as thin lines
PALM_POLY = [0, 1, 5, 9, 13, 17]
# joint radius in scene units, indexed by landmark
RADIUS = [
    0.115,
    0.095,
    0.085,
    0.075,
    0.068,
    0.088,
    0.076,
    0.066,
    0.058,
    0.088,
    0.076,
    0.066,
    0.058,
    0.084,
    0.072,
    0.062,
    0.055,
    0.080,
    0.066,
    0.056,
    0.050,
]
TIPS = (4, 8, 12, 16, 20)
KNUCKLES = (6, 7, 10, 11, 14, 15, 18, 19)
# skin tones for the "human" hand (RGB)
SKIN_TONES = {
    "light": (236, 190, 160),
    "medium": (222, 168, 128),
    "tan": (190, 134, 98),
    "deep": (132, 88, 62),
}
RADIUS = [r * 1.35 for r in RADIUS]  # chunkier, more solid-looking fingers

BODY = (225, 200, 172)  # steel-blue "glass" skin (BGR)
LIGHT_DIR = np.array([-0.45, 0.80, -0.40]) / np.linalg.norm([-0.45, 0.80, -0.40])
GAUGE_STOPS = [(255, 205, 60), (255, 105, 150), (75, 85, 255)]  # azure -> violet -> coral
