"""Shared test helpers: simulated hands and a fake audio device (no camera or speakers needed)."""

from types import SimpleNamespace as NS


def hand(index=True, middle=True, ring=True, pinky=True, ratio=0.6, wrist_x=0.5):
    """Build 21 fake landmarks for an upright hand. Palm size is 0.3."""
    lm = [NS(x=wrist_x, y=0.9) for _ in range(21)]
    xs = {5: 0.40, 9: 0.50, 13: 0.60, 17: 0.70}
    for tip, pip, mcp, up in (
        (8, 6, 5, index),
        (12, 10, 9, middle),
        (16, 14, 13, ring),
        (20, 18, 17, pinky),
    ):
        x = xs[mcp] - 0.5 + wrist_x
        lm[mcp] = NS(x=x, y=0.6)
        lm[pip] = NS(x=x, y=0.5)
        lm[tip] = NS(x=x, y=0.3 if up else 0.68)
    lm[0] = NS(x=wrist_x, y=0.9)
    lm[9] = NS(x=wrist_x, y=0.6)
    # thumb tip placed to the left of the index tip so ratio = distance / 0.3
    lm[4] = NS(x=lm[8].x - ratio * 0.3, y=lm[8].y)
    return lm


class FakeAudio:
    def __init__(self, vol=0.5):
        self.vol, self.mute = vol, False

    def get(self):
        return self.vol

    def set(self, v):
        self.vol = float(v)

    def muted(self):
        return self.mute

    def toggle_mute(self):
        self.mute = not self.mute
        return self.mute


PINCH = dict(index=True, middle=False, ring=False, pinky=False)  # pinch pose: only index extended
