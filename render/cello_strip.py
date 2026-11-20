"""Per-frame cello view: a real CC0 five-string Amati cello, strings bowed.

The plate is a CC0 photograph of the "Amaryllis Fleming" cello from The
Metropolitan Museum of Art (object 898377), background removed and warped so
the measured nut and bridge land on NUT_X/BRIDGE_X and the five strings fan
exactly where the photo's strings were. The photograph supplies the marquetry
fingerboard, pegbox, f-holes, bridge and tailpiece; the strings, finger stops,
bow and glows are drawn on top in local space with real stop geometry (the
speaking length halves every twelve semitones). The whole layer is rotated so
the pegbox reads up-left and the body lower right, with a small badge naming
the sounding note, finger and string.
"""
import math
import queue
import random
import threading
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFilter

from config import FPS, H, W
from render.scene import load_font, make_spark_sprite, palette
from render.score_layer import PANEL_H, PANEL_W, PANEL_X, PANEL_Y

# High E down to low C: open pitches, ordered top-to-bottom in local space.
# The photographed instrument carries five strings (C G D A E), so the top
# string is the high E a fourth above A.
OPEN_PITCHES = (64, 57, 50, 43, 36)
STRING_NAMES = ("E", "A", "D", "G", "C")

# The two bass strings are silver-wound; the three upper strings are gut.
STRING_COLORS = ((236, 230, 212), (228, 218, 192), (216, 200, 166),
                 (204, 210, 216), (208, 214, 220))
STRING_THICK = (
    ((-1, 0.55), (0, 1.0), (1, 0.50)),
    ((-2, 0.35), (-1, 0.62), (0, 1.0), (1, 0.58), (2, 0.32)),
    ((-2, 0.38), (-1, 0.66), (0, 1.0), (1, 0.62), (2, 0.36)),
    ((-3, 0.30), (-2, 0.50), (-1, 0.74), (0, 1.0), (1, 0.70), (2, 0.44), (3, 0.28)),
    ((-3, 0.32), (-2, 0.52), (-1, 0.76), (0, 1.0), (1, 0.72), (2, 0.46), (3, 0.30)),
)
VIB_AMP = (1.3, 1.5, 1.8, 2.2, 2.6)
VIB_WL = (74.0, 84.0, 96.0, 112.0, 130.0)
VIB_OMEGA = (6.2, 5.8, 5.3, 4.8, 4.3)

MID = 592.0
LOCAL_W, LOCAL_H = 3400, 1200
NUT_X, BRIDGE_X = 800, 2400
MAX_FRET = 19            # last stop the drawn geometry still supports
TARGET_FRET = 7          # stops favour the middle positions, like a player
BOW_X = 2245.0
BOW_X_MIN, BOW_X_MAX = 2145.0, 2335.0
BOW_EXCURSION = 40.0
BOW_ANGLE_RANGE = 14

# The photo is upright: pegbox at the top, bridge lower down.
BODY_PHOTO = Path(__file__).resolve().parents[1] / "assets" / "cello_body.jpg"
PHOTO_NUT = (1492.0, 787.0)
PHOTO_SADDLE = (1493.0, 2746.0)
PHOTO_AXIS_U = (0.00051, 0.9999999)   # along the strings, nut -> bridge
PHOTO_AXIS_B = (-1.0, 0.0)            # across them, toward the high E
PHOTO_NUT_XS = (1575.0, 1534.0, 1493.0, 1452.0, 1411.0)
PHOTO_SADDLE_XS = (1585.0, 1539.0, 1493.0, 1447.0, 1401.0)
PHOTO_STRING_W = (6, 7, 8, 9, 10)
PHOTO_STRING_Y0, PHOTO_STRING_Y1 = 758.0, 2772.0
# The instrument rests on a museum mount; everything below the tailpiece
# (mount, floor shadow) is trimmed from the cut-out.
PHOTO_TRIM_Y = 3708

# Whole-layer placement: clockwise tilt, pegbox up-left, body lower right.
TILT_DEG = 36.0
TILT_SCALE = 0.38
ANCHOR_LOCAL = (float(NUT_X), MID)
ANCHOR_SCREEN = (330.0, 242.0)

NOTE_NAMES = ("C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B")
FINGER_NAMES = ("1st", "2nd", "3rd", "4th")

_NUT_V = np.array(PHOTO_NUT, dtype=np.float64)
_U_V = np.array(PHOTO_AXIS_U, dtype=np.float64)
_B_V = np.array(PHOTO_AXIS_B, dtype=np.float64)
_K = (BRIDGE_X - NUT_X) / float(
    np.linalg.norm(np.array(PHOTO_SADDLE, dtype=np.float64) - _NUT_V))


def _across(x_photo, y_photo):
    p = np.array((x_photo, y_photo), dtype=np.float64) - _NUT_V
    return float(MID + _K * float(_B_V @ p))


NUT_Y = tuple(_across(x, PHOTO_NUT[1]) for x in PHOTO_NUT_XS)
SADDLE_Y = tuple(_across(x, PHOTO_SADDLE[1]) for x in PHOTO_SADDLE_XS)


def _string_index(pitch):
    """Highest open string at or below the pitch, as a player would choose."""
    for i in range(5):
        if pitch >= OPEN_PITCHES[i]:
            return i
    return 4


def _stop_x(pitch, si):
    """Fretless stop position: one octave up is exactly half the string."""
    semis = max(0, pitch - OPEN_PITCHES[si])
    return BRIDGE_X - (BRIDGE_X - NUT_X) * (2.0 ** (-semis / 12.0))


def _string_y(si, x):
    frac = (x - NUT_X) / float(BRIDGE_X - NUT_X)
    return NUT_Y[si] + (SADDLE_Y[si] - NUT_Y[si]) * frac


def _photo_affine():
    s = 2
    return s * np.array([
        [_K * _U_V[0], _K * _U_V[1], NUT_X - _K * float(_U_V @ _NUT_V)],
        [_K * _B_V[0], _K * _B_V[1], MID - _K * float(_B_V @ _NUT_V)],
    ])


def _photo_mask(img):
    """Silhouette of the instrument against the neutral studio backdrop."""
    h, w = img.shape[:2]
    mn = img.min(axis=2).astype(np.int16)
    mx = img.max(axis=2).astype(np.int16)
    backdrop = ((mx - mn) <= 26) & (mx >= 80)
    region = backdrop.astype(np.uint8)
    ffmask = np.zeros((h + 2, w + 2), np.uint8)
    step = 6
    seeds = [(x, 0) for x in range(0, w, step)]
    seeds += [(x, h - 1) for x in range(0, w, step)]
    seeds += [(0, y) for y in range(0, h, step)]
    seeds += [(w - 1, y) for y in range(0, h, step)]
    for sx, sy in seeds:
        if region[sy, sx] == 1:
            cv2.floodFill(region, ffmask, (sx, sy), 2)
    fg = (region != 2).astype(np.uint8)
    fg = cv2.morphologyEx(fg, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))
    fg = cv2.morphologyEx(fg, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
    fg = cv2.morphologyEx(fg, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    n, lab, stats, _ = cv2.connectedComponentsWithStats(fg, 8)
    if n > 1:
        keep = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
        fg = (lab == keep).astype(np.uint8)
    fg = _refine_mask(img, fg)
    fg[PHOTO_TRIM_Y:, :] = 0
    fg = (cv2.GaussianBlur(fg.astype(np.float32), (0, 0), 3.0) > 0.5).astype(np.uint8)
    n, lab, stats, _ = cv2.connectedComponentsWithStats(fg, 8)
    if n > 1:
        keep = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
        fg = (lab == keep).astype(np.uint8)
    return fg


def _refine_mask(img, fg):
    """GrabCut strips the cast shadow the colour test leaves on the backdrop,
    then the contour is smoothed so the cut-out edge reads as a clean line."""
    h, w = fg.shape
    S = 0.5
    small = cv2.resize(img, (int(w * S), int(h * S)), interpolation=cv2.INTER_AREA)
    guess = cv2.resize(fg, (int(w * S), int(h * S)), interpolation=cv2.INTER_NEAREST)

    gc = np.where(guess > 0, cv2.GC_PR_FGD, cv2.GC_PR_BGD).astype(np.uint8)
    border = 12
    gc[:border, :] = gc[-border:, :] = cv2.GC_BGD
    gc[:, :border] = gc[:, -border:] = cv2.GC_BGD
    gc[cv2.erode(guess, np.ones((31, 31), np.uint8)) > 0] = cv2.GC_FGD
    bgd = np.zeros((1, 65), np.float64)
    fgd = np.zeros((1, 65), np.float64)
    cv2.grabCut(small, gc, None, bgd, fgd, 4, cv2.GC_INIT_WITH_MASK)
    fg_s = np.where((gc == cv2.GC_FGD) | (gc == cv2.GC_PR_FGD), 1, 0).astype(np.uint8)

    fg_s = cv2.morphologyEx(fg_s, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
    fg_s = (cv2.GaussianBlur(fg_s.astype(np.float32), (0, 0), 2.5) > 0.5).astype(np.uint8)
    n, lab, stats, _ = cv2.connectedComponentsWithStats(fg_s, 8)
    if n > 1:
        fg_s = (lab == 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))).astype(np.uint8)
    fg_s = cv2.morphologyEx(fg_s, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    return cv2.resize(fg_s, (w, h), interpolation=cv2.INTER_NEAREST)


_BODY_CACHE = {}


def _body_photo():
    """The CC0 photo cut out, strings inpainted, warped into local space."""
    cached = _BODY_CACHE.get("warp")
    if cached is not None:
        return cached
    img = cv2.imread(str(BODY_PHOTO), cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"cello body photo missing: {BODY_PHOTO}")
    img = img[:, :, ::-1]
    fg = _photo_mask(img)

    cut = np.zeros(img.shape[:2], np.uint8)
    for xn, xs, width in zip(PHOTO_NUT_XS, PHOTO_SADDLE_XS, PHOTO_STRING_W):
        cv2.line(cut, (int(round(xn)), int(PHOTO_STRING_Y0)),
                 (int(round(xs)), int(PHOTO_STRING_Y1)), 255, width + 2)
    img = cv2.inpaint(img, cut, 4, cv2.INPAINT_TELEA)

    size = (LOCAL_W * 2, LOCAL_H * 2)
    fwd = _photo_affine()
    rgb = cv2.warpAffine(img, fwd, size, flags=cv2.INTER_CUBIC,
                         borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))
    alpha = cv2.warpAffine(fg * 255, fwd, size, flags=cv2.INTER_LINEAR,
                           borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    alpha = cv2.GaussianBlur(alpha, (0, 0), 2.2)

    rgb = cv2.resize(rgb.astype(np.float32) * 0.78, (LOCAL_W, LOCAL_H),
                     interpolation=cv2.INTER_AREA)
    alpha = cv2.resize(alpha, (LOCAL_W, LOCAL_H), interpolation=cv2.INTER_AREA)
    out = np.dstack([np.clip(rgb, 0, 255).astype(np.uint8), alpha])
    _BODY_CACHE["warp"] = out
    return out


def make_cello_base(palette_name):
    pal = palette(palette_name)
    accent = np.array(pal["accent"], dtype=np.float32)
    canvas = _body_photo().copy()
    canvas[:, :, :3] = np.clip(
        canvas[:, :, :3].astype(np.float32) + accent[None, None, :] * 0.022,
        0, 255).astype(np.uint8)

    img = Image.fromarray(canvas, "RGBA")
    blur = np.array(img.getchannel("A").filter(ImageFilter.GaussianBlur(24)),
                    dtype=np.float32)
    shifted = np.zeros_like(blur)
    dx, dy = 8, 16
    shifted[dy:, dx:] = blur[:LOCAL_H - dy, :LOCAL_W - dx]
    shadow = Image.new("RGBA", (LOCAL_W, LOCAL_H), (0, 0, 0, 0))
    shadow.putalpha(Image.fromarray(np.clip(shifted * 0.55, 0, 255).astype(np.uint8)))
    return np.array(Image.alpha_composite(shadow, img), dtype=np.uint8)


_BOW_CACHE = None


def _make_bow_sprites():
    """Pre-rendered cello bow at each tilt angle, composited per frame."""
    global _BOW_CACHE
    if _BOW_CACHE is not None:
        return _BOW_CACHE
    S = 2
    w, h = 380 * S, 1080 * S
    sprites = []
    for ang in range(-BOW_ANGLE_RANGE, BOW_ANGLE_RANGE + 1):
        img = Image.new("RGBA", (w, h), (0, 0, 0, 0))
        d = ImageDraw.Draw(img)
        cx, cy = w // 2, h // 2
        half = 500 * S
        dx = int(math.tan(math.radians(ang)) * half)
        x0, y0 = cx - dx, cy - half
        x1, y1 = cx + dx, cy + half
        off = 36 * S
        # Hair sits on the string side of the stick, with a dark backing edge
        # so the ribbon reads over pale and dark wood alike.
        d.line([(x0 - off - 2 * S, y0 + 10 * S), (x1 - off - 2 * S, y1 - 10 * S)],
               fill=(30, 20, 12, 200), width=16 * S)
        d.line([(x0 - off, y0 + 12 * S), (x1 - off, y1 - 12 * S)],
               fill=(250, 247, 240, 245), width=12 * S)
        d.line([(x0 - off - 2 * S, y0 + 18 * S), (x1 - off - 2 * S, y1 - 18 * S)],
               fill=(255, 253, 248, 175), width=3 * S)
        # Pernambuco stick: dark outline, body, lit edge.
        d.line([(x0, y0), (x1, y1)], fill=(26, 17, 12, 255), width=30 * S)
        d.line([(x0, y0), (x1, y1)], fill=(172, 112, 62, 255), width=24 * S)
        d.line([(x0 + 5 * S, y0), (x1 + 5 * S, y1)], fill=(240, 192, 130, 255), width=7 * S)
        d.line([(x1 - 1 * S, y1 - 130 * S), (x1 - 1 * S, y1 - 5 * S)],
               fill=(50, 34, 24, 255), width=40 * S)
        d.line([(x0 + 1 * S, y0 + 4 * S), (x0 + 1 * S, y0 + 96 * S)],
               fill=(240, 228, 202, 255), width=12 * S)
        arr = np.asarray(img.resize((380, 1080), Image.Resampling.LANCZOS),
                         dtype=np.float32)
        sprites.append((arr[:, :, :3], arr[:, :, 3] / 255.0))
    _BOW_CACHE = sprites
    return sprites


def _clip_add(roi, add):
    roi[:] = np.minimum(roi.astype(np.uint16) + add.astype(np.uint16), 255).astype(np.uint8)


def _glow_sprite(size):
    yy, xx = np.mgrid[0:size, 0:size].astype(np.float32)
    c = (size - 1) / 2.0
    r = np.sqrt((xx - c) ** 2 + (yy - c) ** 2) / c
    return np.clip(1.0 - r, 0, 1) ** 2.6


class CelloStripGenerator:
    def __init__(self, section, notes, seed=0):
        pal = palette(section["palette"])
        self.base = make_cello_base(section["palette"])
        self.key_color = np.array(pal["key_glow"], dtype=np.float32)
        self.accent = np.array(pal["accent"], dtype=np.float32)
        self.finger_glow = _glow_sprite(34)
        self.contact_glow = _glow_sprite(26)
        self.spark = make_spark_sprite(14)
        self.bow_sprites = _make_bow_sprites()
        glow_img = Image.fromarray((_glow_sprite(96) * 255).astype(np.uint8), "L")
        self.body_glow = (np.asarray(glow_img.resize((900, 460), Image.Resampling.BICUBIC),
                                     dtype=np.float32) / 255.0)
        self.M = cv2.getRotationMatrix2D(ANCHOR_LOCAL, -TILT_DEG, TILT_SCALE)
        self.M[0, 2] += ANCHOR_SCREEN[0] - ANCHOR_LOCAL[0]
        self.M[1, 2] += ANCHOR_SCREEN[1] - ANCHOR_LOCAL[1]

        self.badge_sprites = {}
        self.badge_pitch = None
        self.badge_si = 0
        self.badge_alpha = 0.0
        self.badge_since = -10.0
        self.badge_hold = 0.4

        self.section = section
        self.rng = random.Random(seed * 7919 + section["index"] * 104729)

        sec_start, sec_end = section["start"], section["end"]
        self.notes = [n for n in notes if n["start"] <= sec_end and n["end"] >= sec_start - 0.4]
        self.cello_notes = [n for n in self.notes if n["hand"] == "R"]
        self.active = []
        self.next_idx = 0
        self.particles = []

        self.finger_x = [None] * 5
        self.finger_level = [0.0] * 5
        self.stop_anchor = [float(NUT_X)] * 5
        self.vib = [0.0] * 5

        self.bow_dir = 1.0
        self.bow_tilt = 0.0
        self.bow_y = 0.0
        self.bow_x = float(BOW_X)
        self.bow_target_x = float(BOW_X)
        self.stroke_t0 = -10.0
        self.stroke_dur = 0.0

        self.frame_events = {}
        for n in self.cello_notes:
            f = int(round((n["start"] - sec_start) * FPS))
            if 0 <= f < section["frames"]:
                self.frame_events.setdefault(f, []).append(n)

    def frames(self):
        sec = self.section
        frames = sec["frames"]
        dt = 1.0 / FPS
        for i in range(frames):
            t = sec["start"] + i * dt
            layer = self.base.copy()
            cello, piano = self._pressed(t)
            self._on_onsets(i, t)
            self._draw_strings(layer, t)
            self._draw_stops(layer, t, cello)
            self._update_bow(t, cello)
            self._draw_bow(layer, t, cello)
            self._draw_body(layer, piano)
            self._update_particles(layer, dt)
            frame = cv2.warpAffine(layer, self.M, (W, H), flags=cv2.INTER_LINEAR,
                                   borderMode=cv2.BORDER_CONSTANT,
                                   borderValue=(0, 0, 0, 0))
            self._update_badge(t, cello)
            self._draw_badge(frame)
            yield frame.tobytes()

    def _pressed(self, t):
        while self.next_idx < len(self.notes) and self.notes[self.next_idx]["start"] <= t:
            self.active.append(self.notes[self.next_idx])
            self.next_idx += 1
        self.active = [n for n in self.active if n["end"] + 0.25 >= t]
        cello = {}
        piano = 0.0
        for n in self.active:
            if t < n["start"]:
                continue
            rel = t - n["start"]
            attack = min(1.0, rel / 0.012)
            sustain = 0.5 + 0.5 * math.exp(-rel / 0.55)
            value = attack * sustain * (n["vel"] / 127.0)
            if t > n["end"]:
                value *= math.exp(-(t - n["end"]) / 0.13)
            if n["hand"] == "R":
                prev = cello.get(n["pitch"])
                if prev is None or value > prev[0]:
                    cello[n["pitch"]] = (value, bool(n.get("arp", False)))
            else:
                piano = max(piano, value)
        return cello, piano

    def _on_onsets(self, frame_index, t):
        for n in self.frame_events.get(frame_index, ()):
            self.bow_dir = -self.bow_dir
            self.stroke_t0 = t
            self.stroke_dur = max(0.35, min(2.2, n["end"] - n["start"]))
            si = _string_index(n["pitch"])
            self.vib[si] = min(1.35, self.vib[si] + 0.5 + n["vel"] / 240.0)
            self._spawn(n, si)

    def _draw_strings(self, layer, t):
        xs = np.arange(NUT_X - 24, BRIDGE_X + 4)
        span = float(BRIDGE_X - NUT_X)
        frac = (xs - NUT_X) / span
        for si in range(4, -1, -1):
            y = NUT_Y[si] + (SADDLE_Y[si] - NUT_Y[si]) * frac
            vib = self.vib[si]
            if vib > 0.015:
                xa = max(float(NUT_X), self.stop_anchor[si])
                u = np.clip((xs - xa) / max(1.0, BRIDGE_X - xa), 0.0, 1.0)
                env = np.sin(np.pi * u)
                y = y + (VIB_AMP[si] * vib * env
                         * np.sin(2 * np.pi * (xs - xa) / VIB_WL[si]
                                  - t * VIB_OMEGA[si]))
            yi = np.clip(np.round(y).astype(np.int32), 6, LOCAL_H - 7)
            col = np.array(STRING_COLORS[si], dtype=np.float32) * (1.0 + 0.45 * min(1.0, vib))
            for dy, mul in STRING_THICK[si]:
                rows = np.clip(yi + dy, 0, LOCAL_H - 1)
                layer[rows, xs, :3] = np.clip(col * mul, 0, 255).astype(np.uint8)
                layer[rows, xs, 3] = 255

    def _draw_stops(self, layer, t, cello):
        string_val = [0.0] * 5
        string_pitch = [None] * 5
        string_arp = [False] * 5
        for pitch, (val, arp) in cello.items():
            si = _string_index(pitch)
            if val > string_val[si]:
                string_val[si] = val
                string_pitch[si] = pitch
                string_arp[si] = arp

        for si in range(5):
            self.finger_level[si] = max(string_val[si], self.finger_level[si] * 0.90)
            pitch = string_pitch[si]
            if pitch is not None:
                anchor = _stop_x(pitch, si)
                self.stop_anchor[si] = anchor
                if self.finger_x[si] is None:
                    self.finger_x[si] = anchor
                else:
                    self.finger_x[si] += (anchor - self.finger_x[si]) * 0.35

            fx = self.finger_x[si]
            lvl = self.finger_level[si]
            if fx is None or lvl <= 0.015:
                continue
            y = _string_y(si, fx)
            open_str = pitch is not None and pitch <= OPEN_PITCHES[si]
            glow = (0.5 if open_str else 1.0) * (0.72 if string_arp[si] else 1.0)

            self._add_column(layer, int(round(fx)), int(round(y - 32)), int(round(y - 4)),
                             self.key_color, (0.06 + 0.28 * lvl) * glow)
            self._stamp(layer, fx, y, self.finger_glow, self.key_color, 0.42 * lvl * glow)
            self._stamp(layer, fx, y, self.spark, self.accent, 0.55 * lvl * glow)

            x0 = max(0, int(fx) - 4)
            x1 = min(LOCAL_W, int(fx) + 4)
            band = np.broadcast_to((self.accent * lvl * 0.55 * glow).astype(np.uint16),
                                   (3, x1 - x0, 3))
            _clip_add(layer[int(y) - 1:int(y) + 2, x0:x1, :3], band)

            if not open_str and lvl > 0.02:
                self._draw_wave(layer, si, fx, t, lvl)
            self.vib[si] = max(string_val[si], self.vib[si] * 0.94)

            # Light where the bow hair meets the string.
            if string_val[si] > 0.02:
                self._stamp(layer, int(round(self.bow_x)) - 34, y,
                            self.contact_glow, self.key_color, 0.26 * string_val[si])

    def _draw_wave(self, layer, si, xa, t, energy):
        xb = float(BRIDGE_X)
        x0 = int(xa) + 6
        if xb - x0 < 12:
            return
        xs = np.arange(x0, int(xb))
        u = (xs - xa) / max(1.0, xb - xa)
        env = np.sin(np.pi * u)
        y = _string_y(si, xs) + (VIB_AMP[si] + 0.6) * energy * env \
            * np.sin(2 * np.pi * (xs - xa) / VIB_WL[si] - t * VIB_OMEGA[si])
        yi = np.clip(np.round(y).astype(np.int32), 0, LOCAL_H - 1)
        col = np.clip(self.accent * (0.35 + 0.5 * energy), 0, 255).astype(np.uint16)
        cur = layer[yi, xs, :3].astype(np.uint16)
        layer[yi, xs, :3] = np.minimum(cur + col[None, :], 255).astype(np.uint8)

    def _update_bow(self, t, cello):
        energy = max((v for v, _ in cello.values()), default=0.0)
        if cello:
            # The bow tracks the melody: low stops draw it nearer the board,
            # high stops nearer the bridge, as a player shifts contact point.
            pitch = max(cello, key=lambda p: cello[p][0])
            target = 2185.0 + (pitch - 57) * 4.2
            self.bow_target_x = max(BOW_X_MIN, min(BOW_X_MAX, target))
        self.bow_x += (self.bow_target_x - self.bow_x) * 0.08
        if self.stroke_dur > 0:
            phase = min(1.0, max(0.0, (t - self.stroke_t0) / self.stroke_dur))
            target = self.bow_dir * (phase * 2.0 - 1.0) * BOW_EXCURSION
        else:
            target = 0.0
        self.bow_y += (target - self.bow_y) * 0.35
        sway = 5.0 * math.sin(t * 2.1) + 2.5 * math.sin(t * 5.3)
        self.bow_tilt = (-8.0 * self.bow_dir * (0.35 + 0.65 * min(1.0, energy))
                         + sway * 0.45)

    def _draw_bow(self, strip, t, cello):
        energy = min(1.0, max((v for v, _ in cello.values()), default=0.0))
        idx = max(0, min(2 * BOW_ANGLE_RANGE,
                         int(round(self.bow_tilt)) + BOW_ANGLE_RANGE))
        rgb, alpha = self.bow_sprites[idx]
        # The bow stays readable while it rests: a player holds it above the
        # strings between phrases, it does not vanish.
        a_mul = 0.62 + 0.38 * energy
        sh, sw = rgb.shape[0], rgb.shape[1]
        ox = int(round(self.bow_x - sw * 0.5 + 6 * math.sin(t * 1.1)))
        oy = int(round(MID - sh * 0.5 + self.bow_y))
        cx0, cy0 = max(0, ox), max(0, oy)
        cx1, cy1 = min(LOCAL_W, ox + sw), min(LOCAL_H, oy + sh)
        if cx1 <= cx0 or cy1 <= cy0:
            return
        sub_rgb = rgb[cy0 - oy:cy1 - oy, cx0 - ox:cx1 - ox]
        sub_a = (alpha[cy0 - oy:cy1 - oy, cx0 - ox:cx1 - ox] * a_mul)[:, :, None]
        roi = strip[cy0:cy1, cx0:cx1]
        roi[:, :, :3] = np.clip(roi[:, :, :3] * (1.0 - sub_a) + sub_rgb * sub_a,
                                0, 255).astype(np.uint8)
        roi[:, :, 3] = np.maximum(roi[:, :, 3], (sub_a[:, :, 0] * 255).astype(np.uint8))

    def _draw_body(self, strip, piano):
        if piano <= 0.01:
            return
        sh, sw = self.body_glow.shape
        cx, cy = 2260, int(MID)
        x0, y0 = cx - sw // 2, cy - sh // 2
        cy0, cy1 = max(0, y0), min(LOCAL_H, y0 + sh)
        cx0, cx1 = max(0, x0), min(LOCAL_W, x0 + sw)
        if cy1 <= cy0 or cx1 <= cx0:
            return
        sub = self.body_glow[cy0 - y0:cy1 - y0, cx0 - x0:cx1 - x0][:, :, None]
        add = (sub * self.accent[None, None, :] * (0.16 * piano)).astype(np.uint16)
        _clip_add(strip[cy0:cy1, cx0:cx1, :3], add)

    def _add_column(self, strip, cx, y0, y1, color, intensity):
        h = y1 - y0
        if h <= 2:
            return
        w = 22
        x0 = max(0, cx - w // 2)
        x1 = min(LOCAL_W, x0 + w)
        if x1 <= x0:
            return
        xx = np.arange(x1 - x0, dtype=np.float32) + (x0 - (cx - w // 2))
        horiz = np.clip(1.0 - np.abs(xx - w / 2) / (w / 2), 0, 1)[None, :, None] ** 1.4
        vert = np.linspace(0.35, 1.0, h, dtype=np.float32)[:, None, None]
        add = color[None, None, :] * (horiz * vert) * intensity
        _clip_add(strip[y0:y1, x0:x1, :3], np.clip(add, 0, 255).astype(np.uint16))

    def _stamp(self, strip, cx, cy, sprite, color, intensity):
        sh = sprite.shape[0]
        x0, y0 = int(round(cx - sh / 2)), int(round(cy - sh / 2))
        cx0, cy0 = max(0, x0), max(0, y0)
        cx1, cy1 = min(LOCAL_W, x0 + sh), min(LOCAL_H, y0 + sh)
        if cx1 <= cx0 or cy1 <= cy0:
            return
        sub = sprite[cy0 - y0:cy1 - y0, cx0 - x0:cx1 - x0][:, :, None]
        add = (sub * color[None, None, :] * (255.0 * min(1.0, intensity))).astype(np.uint16)
        _clip_add(strip[cy0:cy1, cx0:cx1, :3], add)

    def _blend(self, frame, x, y, rgb, alpha, mul=1.0):
        sh, sw = alpha.shape
        cx0, cy0 = max(0, x), max(0, y)
        cx1, cy1 = min(W, x + sw), min(H, y + sh)
        if cx1 <= cx0 or cy1 <= cy0:
            return
        a = (alpha[cy0 - y:cy1 - y, cx0 - x:cx1 - x] * mul)[:, :, None]
        roi = frame[cy0:cy1, cx0:cx1]
        roi[:, :, :3] = np.clip(roi[:, :, :3] * (1.0 - a)
                                + rgb[cy0 - y:cy1 - y, cx0 - x:cx1 - x] * a,
                                0, 255).astype(np.uint8)
        roi[:, :, 3] = np.maximum(roi[:, :, 3], (a[:, :, 0] * 255).astype(np.uint8))

    def _badge_sprite(self, pitch, si):
        key = (pitch, si)
        sprite = self.badge_sprites.get(key)
        if sprite is not None:
            return sprite
        note = f"{NOTE_NAMES[pitch % 12]}{pitch // 12 - 1}"
        semis = pitch - OPEN_PITCHES[si]
        if semis <= 0:
            detail = "open string"
        else:
            # Whole-step hand frame: each finger sits two semitones above the
            # previous one, e.g. on the D string E=1st, F/F#=2nd, G/G#=3rd.
            finger = min(4, max(1, (semis + 1) // 2))
            detail = (f"{FINGER_NAMES[finger - 1]} finger · "
                      f"{STRING_NAMES[si]} string")
        f_note = load_font("sans", 14)
        f_detail = load_font("sans", 11)
        probe = ImageDraw.Draw(Image.new("RGBA", (8, 8)))
        nw = int(probe.textlength(note, font=f_note))
        dw = int(probe.textlength(detail, font=f_detail))
        pad_x, gap, h = 10, 8, 29
        w = pad_x * 2 + nw + gap + dw
        img = Image.new("RGBA", (w, h), (0, 0, 0, 0))
        d = ImageDraw.Draw(img)
        d.rounded_rectangle((0, 0, w - 1, h - 1), radius=8, fill=(7, 10, 17, 214),
                            outline=tuple(int(c) for c in self.accent) + (105,), width=1)
        d.text((pad_x, 6), note, font=f_note, fill=(236, 243, 255, 240))
        d.text((pad_x + nw + gap, 9), detail, font=f_detail,
               fill=(204, 214, 232, 225))
        arr = np.asarray(img, dtype=np.float32)
        sprite = (arr[:, :, :3], arr[:, :, 3] / 255.0, w, h)
        self.badge_sprites[key] = sprite
        return sprite

    def _update_badge(self, t, cello):
        if cello:
            pitch = max(cello, key=lambda p: cello[p][0])
            if pitch != self.badge_pitch and t - self.badge_since >= self.badge_hold:
                self.badge_pitch = pitch
                self.badge_si = _string_index(pitch)
                self.badge_since = t
            target = 1.0
        else:
            target = 0.0
        self.badge_alpha += (target - self.badge_alpha) * 0.22

    def _draw_badge(self, frame):
        if self.badge_pitch is None or self.badge_alpha < 0.03:
            return
        rgb, alpha, w, h = self._badge_sprite(self.badge_pitch, self.badge_si)
        fx = self.finger_x[self.badge_si]
        if fx is None:
            fx = _stop_x(self.badge_pitch, self.badge_si)
        sx, sy = self._project(fx, _string_y(self.badge_si, fx))
        x = int(round(sx + 18))
        y = int(round(sy + 16))
        if (x + w > PANEL_X - 6 and x < PANEL_X + PANEL_W + 6
                and y + h > PANEL_Y - 6 and y < PANEL_Y + PANEL_H + 6):
            x = int(round(sx - w - 18))
        x = max(16, min(W - w - 16, x))
        y = max(16, min(H - h - 16, y))
        self._blend(frame, x, y, rgb, alpha, min(1.0, self.badge_alpha))

    def _project(self, x, y):
        return (self.M[0, 0] * x + self.M[0, 1] * y + self.M[0, 2],
                self.M[1, 0] * x + self.M[1, 1] * y + self.M[1, 2])

    def _spawn(self, note, si):
        count = 2 + int(note["vel"] / 60)
        y = _string_y(si, self.bow_x)
        for _ in range(count):
            self.particles.append({
                "x": self.bow_x + self.rng.uniform(-10, 10),
                "y": y + self.rng.uniform(-4, 8),
                "vx": self.rng.uniform(-26, 26),
                "vy": -self.rng.uniform(40, 110),
                "life": self.rng.uniform(0.4, 0.85),
                "age": 0.0,
            })

    def _update_particles(self, strip, dt):
        alive = []
        sprite = self.spark
        sh = sprite.shape[0]
        for p in self.particles:
            p["age"] += dt
            if p["age"] >= p["life"]:
                continue
            p["x"] += p["vx"] * dt
            p["vy"] += 30 * dt
            p["y"] += p["vy"] * dt
            life = 1.0 - p["age"] / p["life"]
            x, y = int(p["x"]), int(p["y"])
            x0, y0 = x - sh // 2, y - sh // 2
            x1, y1 = x0 + sh, y0 + sh
            if x1 <= 0 or y1 <= 0 or x0 >= LOCAL_W or y0 >= LOCAL_H:
                continue
            cx0, cy0 = max(0, x0), max(0, y0)
            cx1, cy1 = min(LOCAL_W, x1), min(LOCAL_H, y1)
            sub = sprite[cy0 - y0:cy1 - y0, cx0 - x0:cx1 - x0]
            add = (sub[:, :, None] * self.accent[None, None, :]
                   * (life ** 1.4) * 190).astype(np.uint16)
            _clip_add(strip[cy0:cy1, cx0:cx1, :3], add)
            alive.append(p)
        self.particles = alive


class PipeWriter:
    """Writes frames to a subprocess stdin on a background thread."""

    def __init__(self, stream, maxsize=4):
        self.stream = stream
        self.q = queue.Queue(maxsize=maxsize)
        self.error = None
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def write(self, data):
        while not self.error:
            try:
                self.q.put(data, timeout=0.5)
                return
            except queue.Full:
                continue

    def close(self):
        while True:
            try:
                self.q.put(None, timeout=0.5)
                break
            except queue.Full:
                continue
        self.thread.join()

    def _run(self):
        while True:
            item = self.q.get()
            if item is None:
                break
            if self.error is not None:
                continue
            try:
                self.stream.write(item)
            except (BrokenPipeError, ValueError, OSError) as exc:
                self.error = exc


def stream_section(section, notes, stream, seed=0):
    gen = CelloStripGenerator(section, notes, seed)
    writer = PipeWriter(stream)
    try:
        for frame in gen.frames():
            writer.write(frame)
    finally:
        writer.close()
        try:
            stream.close()
        except Exception:
            pass
