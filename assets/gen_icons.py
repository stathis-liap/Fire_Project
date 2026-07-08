"""gen_icons.py — Generates minimal vector-style PNG icons for the WILSON UI.

Draws each icon at 4x supersample resolution with PIL primitives (lines,
circles, polygons, arcs) then downsamples for anti-aliasing, producing
clean, consistent, small monochrome/two-tone icons to replace emoji.

Run once: python assets/gen_icons.py
Outputs to assets/icons/*.png
"""
import math
import os
from PIL import Image, ImageDraw

OUT_DIR = os.path.join(os.path.dirname(__file__), "icons")
os.makedirs(OUT_DIR, exist_ok=True)

S = 256          # supersample canvas size
FINAL = 64        # final output size
W = 15            # stroke width at supersample scale

# Catppuccin Mocha palette (matches existing UI colors)
TEXT   = (205, 214, 244, 255)   # #cdd6f4 — avoid for icons: too light for a light theme
DIM    = (108, 112, 134, 255)   # #6c7086
NEUTRAL = (147, 153, 178, 255)  # #9399b2 — Mocha overlay2: readable on both dark and light bg
BLUE   = (137, 180, 250, 255)   # #89b4fa
TEAL   = (137, 220, 235, 255)   # #89dceb
GREEN  = (166, 227, 161, 255)   # #a6e3a1
YELLOW = (249, 226, 175, 255)   # #f9e2af
PEACH  = (250, 179, 135, 255)   # #fab387
PINK   = (243, 139, 168, 255)   # #f38ba8
MAUVE  = (203, 166, 247, 255)   # #cba6f7


def canvas():
    return Image.new("RGBA", (S, S), (0, 0, 0, 0))


def stroke(d, pts, color, width=W, closed=False):
    if closed:
        pts = pts + [pts[0]]
    d.line(pts, fill=color, width=width, joint="curve")
    r = width / 2
    for (x, y) in (pts if not closed else pts[:-1]):
        d.ellipse([x - r, y - r, x + r, y + r], fill=color)


def circle(d, cx, cy, r, color, width=W, fill=None):
    if fill:
        d.ellipse([cx - r, cy - r, cx + r, cy + r], fill=fill)
    else:
        d.ellipse([cx - r, cy - r, cx + r, cy + r], outline=color, width=width)


def dot(d, cx, cy, r, color):
    d.ellipse([cx - r, cy - r, cx + r, cy + r], fill=color)


def save(img, name):
    img = img.resize((FINAL, FINAL), Image.LANCZOS)
    img.save(os.path.join(OUT_DIR, name + ".png"))
    print("wrote", name + ".png")


def smooth_closed(pts, per_seg=14):
    """Catmull-Rom spline through pts (closed loop) -> dense smooth polygon."""
    n = len(pts)
    out = []
    for i in range(n):
        p0 = pts[(i - 1) % n]
        p1 = pts[i]
        p2 = pts[(i + 1) % n]
        p3 = pts[(i + 2) % n]
        for t_i in range(per_seg):
            t = t_i / per_seg
            t2, t3 = t * t, t * t * t
            x = 0.5 * ((2 * p1[0]) + (-p0[0] + p2[0]) * t
                       + (2*p0[0] - 5*p1[0] + 4*p2[0] - p3[0]) * t2
                       + (-p0[0] + 3*p1[0] - 3*p2[0] + p3[0]) * t3)
            y = 0.5 * ((2 * p1[1]) + (-p0[1] + p2[1]) * t
                       + (2*p0[1] - 5*p1[1] + 4*p2[1] - p3[1]) * t2
                       + (-p0[1] + 3*p1[1] - 3*p2[1] + p3[1]) * t3)
            out.append((x, y))
    return out


def gradient_polygon(img, poly_pts, color_top, color_bottom):
    """Alpha-composite a vertical gradient clipped to poly_pts onto img."""
    mask = Image.new("L", img.size, 0)
    ImageDraw.Draw(mask).polygon(poly_pts, fill=255)
    ys = [p[1] for p in poly_pts]
    y0, y1 = min(ys), max(ys)
    grad = Image.new("RGBA", img.size, (0, 0, 0, 0))
    gd = ImageDraw.Draw(grad)
    for y in range(int(y0), int(y1) + 1):
        t = (y - y0) / max(y1 - y0, 1)
        r = int(color_top[0] * (1 - t) + color_bottom[0] * t)
        g = int(color_top[1] * (1 - t) + color_bottom[1] * t)
        b = int(color_top[2] * (1 - t) + color_bottom[2] * t)
        gd.line([(0, y), (img.size[0], y)], fill=(r, g, b, 255))
    grad.putalpha(mask)
    img.alpha_composite(grad)


def teardrop_points(cx, cy, rw, rh, rot=0, n=48):
    """Flame/teardrop silhouette via polar formula r = 1 - sin(theta)."""
    pts = []
    for i in range(n):
        t = (i / (n - 1)) * 2 * math.pi
        r = (1 - math.sin(t)) / 2
        x = r * math.cos(t) * rw
        y = r * math.sin(t) * rh
        # rotate
        xr = x * math.cos(rot) - y * math.sin(rot)
        yr = x * math.sin(rot) + y * math.cos(rot)
        pts.append((cx + xr, cy + yr))
    return pts


# ── 1. flame ────────────────────────────────────────────────────────────────
# Outer silhouette: asymmetric emoji-style flame (flatter base, right-leaning
# tip, bigger left bulge) via a Catmull-Rom spline for an organic outline.
_FLAME_OUTER_ANCHORS = [
    (144, 4),     # tip, leaning right
    (178, 54),    # upper right shoulder
    (198, 122),   # widest right bulge
    (180, 180),   # lower right, tucking toward the base
    (142, 222),   # bottom-right (flatter base, not a rounded point)
    (108, 224),   # bottom-left
    (66, 178),    # lower left
    (50, 110),    # widest left bulge (bigger than the right -> lean)
    (90, 52),     # upper left shoulder curving back to the tip
]
_FLAME_RED    = (224, 49, 38, 255)    # deep red, flame base
_FLAME_ORANGE = (255, 171, 63, 255)   # bright orange, flame tip

# Inner core: a "W" (WILSON) traced by the flame-tongue peaks at the top,
# closed off with a rounded belly at the bottom — reads as a stylised flame
# whose inner highlight spells out a W.
_FLAME_INNER_W = [
    (92, 108), (112, 176), (128, 116), (144, 176), (164, 108),
    (176, 166), (158, 208), (128, 226), (98, 208), (80, 166),
]
_FLAME_YELLOW_HI = (255, 224, 120, 255)  # pale hot-yellow, inner tip
_FLAME_YELLOW_LO = (255, 173, 61, 255)   # warm orange-yellow, inner base


def icon_flame():
    img = canvas()
    outer = smooth_closed(_FLAME_OUTER_ANCHORS)
    gradient_polygon(img, outer, _FLAME_ORANGE, _FLAME_RED)
    gradient_polygon(img, _FLAME_INNER_W, _FLAME_YELLOW_HI, _FLAME_YELLOW_LO)
    save(img, "flame")


# ── 2. ash (burned) ──────────────────────────────────────────────────────────
def icon_ash():
    img = canvas(); d = ImageDraw.Draw(img)
    d.pieslice([20, 60, 236, 276], 180, 360, fill=DIM)
    d.line([(20, 168), (236, 168)], fill=DIM, width=W)
    for cx in (86, 128, 170):
        dot(d, cx, 128, 8, (30, 30, 46, 255))
    save(img, "ash")


# ── 3. brain ─────────────────────────────────────────────────────────────────
def icon_brain():
    img = canvas(); d = ImageDraw.Draw(img)
    d.rounded_rectangle([34, 50, 222, 206], radius=78, outline=MAUVE, width=W)
    d.line([(128, 54), (128, 202)], fill=MAUVE, width=W)
    # wrinkle arcs, left hemisphere
    d.arc([56, 78, 116, 126], 200, 350, fill=MAUVE, width=int(W*0.65))
    d.arc([60, 128, 120, 176], 20, 170, fill=MAUVE, width=int(W*0.65))
    # wrinkle arcs, right hemisphere
    d.arc([140, 78, 200, 126], 190, 340, fill=MAUVE, width=int(W*0.65))
    d.arc([136, 128, 196, 176], 10, 160, fill=MAUVE, width=int(W*0.65))
    save(img, "brain")


# ── 4. flask (optimizer) ─────────────────────────────────────────────────────
def icon_flask():
    img = canvas(); d = ImageDraw.Draw(img)
    d.line([(96, 26), (160, 26)], fill=BLUE, width=W)
    d.line([(108, 26), (108, 96)], fill=BLUE, width=W)
    d.line([(148, 26), (148, 96)], fill=BLUE, width=W)
    stroke(d, [(108, 96), (56, 195), (56, 205)], BLUE)
    stroke(d, [(148, 96), (200, 195), (200, 205)], BLUE)
    d.arc([56, 175, 200, 235], 0, 180, fill=BLUE, width=W)
    d.line([(80, 150), (176, 150)], fill=BLUE, width=int(W*0.7))
    save(img, "flask")


# ── 5. bar-chart ─────────────────────────────────────────────────────────────
def icon_barchart():
    img = canvas(); d = ImageDraw.Draw(img)
    d.line([(40, 216), (216, 216)], fill=TEAL, width=W)
    d.line([(70, 216), (70, 110)], fill=TEAL, width=W)
    d.line([(128, 216), (128, 50)], fill=TEAL, width=W)
    d.line([(186, 216), (186, 150)], fill=TEAL, width=W)
    save(img, "bar-chart")


# ── 6. target ─────────────────────────────────────────────────────────────────
def icon_target():
    img = canvas(); d = ImageDraw.Draw(img)
    circle(d, 128, 128, 90, PINK)
    circle(d, 128, 128, 52, PINK)
    dot(d, 128, 128, 14, PINK)
    save(img, "target")


# ── 7. alert-triangle ─────────────────────────────────────────────────────────
def icon_alert():
    img = canvas(); d = ImageDraw.Draw(img)
    stroke(d, [(128, 26), (226, 216), (30, 216)], YELLOW, closed=True)
    d.line([(128, 96), (128, 158)], fill=YELLOW, width=W)
    dot(d, 128, 188, 9, YELLOW)
    save(img, "alert-triangle")


# ── 8. box (3D view) ─────────────────────────────────────────────────────────
def icon_box():
    img = canvas(); d = ImageDraw.Draw(img)
    stroke(d, [(128, 24), (226, 76), (128, 128), (30, 76)], NEUTRAL, closed=True)
    stroke(d, [(30, 76), (30, 180), (128, 232), (128, 128)], NEUTRAL, closed=True)
    stroke(d, [(226, 76), (226, 180), (128, 232)], NEUTRAL)
    save(img, "box")


# ── 9. wind ───────────────────────────────────────────────────────────────────
def icon_wind():
    img = canvas(); d = ImageDraw.Draw(img)
    stroke(d, [(30, 86), (150, 86)], TEAL)
    d.arc([120, 46, 200, 126], -90, 90, fill=TEAL, width=W)
    stroke(d, [(30, 128), (190, 128)], TEAL)
    d.arc([150, 98, 230, 178], -90, 90, fill=TEAL, width=W)
    stroke(d, [(30, 170), (120, 170)], TEAL)
    save(img, "wind")


# ── 10. droplet ───────────────────────────────────────────────────────────────
def icon_droplet():
    img = canvas(); d = ImageDraw.Draw(img)
    pts = teardrop_points(128, 118, 74, 96, rot=math.pi)
    d.polygon(pts, fill=BLUE)
    save(img, "droplet")


# ── 11/12. volume-2 / volume-x ───────────────────────────────────────────────
def icon_volume(muted=False):
    img = canvas(); d = ImageDraw.Draw(img)
    color = DIM if muted else NEUTRAL
    d.polygon([(30, 96), (76, 96), (76, 160), (30, 160)], fill=color)
    d.polygon([(76, 96), (130, 56), (130, 200), (76, 160)], fill=color)
    if muted:
        stroke(d, [(160, 100), (220, 156)], color)
        stroke(d, [(220, 100), (160, 156)], color)
    else:
        d.arc([150, 90, 205, 166], -60, 60, fill=color, width=W)
        d.arc([165, 65, 232, 191], -55, 55, fill=color, width=W)
    save(img, "volume-x" if muted else "volume-2")


# ── 13. wrench (tools) ────────────────────────────────────────────────────────
def icon_wrench():
    img = canvas(); d = ImageDraw.Draw(img)
    circle(d, 176, 80, 40, NEUTRAL, width=W)
    stroke(d, [(148, 108), (60, 196), (40, 216), (60, 216), (80, 196)], NEUTRAL)
    d.line([(196, 60), (216, 40)], fill=NEUTRAL, width=W)
    save(img, "wrench")


# ── 14. bug (debug) ───────────────────────────────────────────────────────────
def icon_bug():
    img = canvas(); d = ImageDraw.Draw(img)
    d.rounded_rectangle([76, 86, 180, 216], radius=48, outline=GREEN, width=W)
    d.arc([92, 60, 164, 112], 180, 360, fill=GREEN, width=W)
    stroke(d, [(70, 60), (92, 82)], GREEN)
    stroke(d, [(186, 60), (164, 82)], GREEN)
    stroke(d, [(20, 128), (76, 128)], GREEN)
    stroke(d, [(180, 128), (236, 128)], GREEN)
    stroke(d, [(30, 210), (76, 190)], GREEN)
    stroke(d, [(226, 210), (180, 190)], GREEN)
    d.line([(128, 96), (128, 216)], fill=GREEN, width=int(W*0.7))
    save(img, "bug")


# ── 15. truck ─────────────────────────────────────────────────────────────────
def icon_truck():
    img = canvas(); d = ImageDraw.Draw(img)
    d.rounded_rectangle([20, 96, 148, 180], radius=8, outline=PINK, width=W)
    stroke(d, [(148, 120), (196, 120), (232, 156), (232, 180), (148, 180)], PINK, closed=False)
    circle(d, 74, 196, 22, PINK, width=W)
    circle(d, 188, 196, 22, PINK, width=W)
    save(img, "truck")


# ── 16. satellite ─────────────────────────────────────────────────────────────
def icon_satellite():
    img = canvas(); d = ImageDraw.Draw(img)
    d.rectangle([96, 96, 160, 160], fill=None, outline=NEUTRAL, width=W)
    d.line([(72, 72), (96, 96)], fill=NEUTRAL, width=W)
    d.line([(184, 184), (160, 160)], fill=NEUTRAL, width=W)
    stroke(d, [(150, 106), (204, 52)], NEUTRAL)
    stroke(d, [(56, 40), (20, 40)], NEUTRAL)
    stroke(d, [(40, 24), (40, 60)], NEUTRAL)
    d.arc([170, 150, 236, 216], 200, 300, fill=NEUTRAL, width=int(W*0.7))
    save(img, "satellite")


# ── 17. leaf (vegetation) ─────────────────────────────────────────────────────
def icon_leaf():
    img = canvas(); d = ImageDraw.Draw(img)
    d.pieslice([26, 26, 230, 230], 200, 20, fill=GREEN)
    stroke(d, [(56, 200), (200, 56)], (30, 30, 46, 255), width=int(W*0.8))
    save(img, "leaf")


# ── 18. arrow-up (wind dir, rotatable) ───────────────────────────────────────
def icon_arrow_up():
    img = canvas(); d = ImageDraw.Draw(img)
    stroke(d, [(128, 226), (128, 34)], BLUE)
    stroke(d, [(70, 96), (128, 30), (186, 96)], BLUE)
    save(img, "arrow-up")


# ── 19. thermometer ───────────────────────────────────────────────────────────
def icon_thermometer():
    img = canvas(); d = ImageDraw.Draw(img)
    d.rounded_rectangle([104, 30, 152, 168], radius=24, outline=PINK, width=W)
    circle(d, 128, 190, 34, PINK, fill=PINK)
    d.line([(128, 168), (128, 190)], fill=PINK, width=int(W*1.4))
    save(img, "thermometer")


# ── 20. zap (risk) ────────────────────────────────────────────────────────────
def icon_zap():
    img = canvas(); d = ImageDraw.Draw(img)
    d.polygon([(140, 20), (56, 140), (112, 140), (100, 236), (200, 108), (140, 108)],
               fill=YELLOW)
    save(img, "zap")


# ── 21. clock ─────────────────────────────────────────────────────────────────
def icon_clock():
    img = canvas(); d = ImageDraw.Draw(img)
    circle(d, 128, 128, 96, NEUTRAL, width=W)
    d.line([(128, 128), (128, 72)], fill=NEUTRAL, width=W)
    d.line([(128, 128), (172, 150)], fill=NEUTRAL, width=W)
    save(img, "clock")


# ── 22. cloud-sun (weather) ───────────────────────────────────────────────────
def icon_cloud_sun():
    img = canvas(); d = ImageDraw.Draw(img)
    circle(d, 74, 66, 26, YELLOW, fill=YELLOW)
    for ang in range(0, 360, 45):
        x1 = 74 + 36 * math.cos(math.radians(ang)); y1 = 66 + 36 * math.sin(math.radians(ang))
        x2 = 74 + 48 * math.cos(math.radians(ang)); y2 = 66 + 48 * math.sin(math.radians(ang))
        d.line([(x1, y1), (x2, y2)], fill=YELLOW, width=int(W*0.6))
    # Solid filled cloud: base pill + two puffs, all one solid colour
    circle(d, 148, 128, 44, NEUTRAL, fill=NEUTRAL)
    circle(d, 98, 144, 34, NEUTRAL, fill=NEUTRAL)
    d.rounded_rectangle([56, 144, 214, 210], radius=32, fill=NEUTRAL)
    save(img, "cloud-sun")


# ── 23. clipboard-list (events) ───────────────────────────────────────────────
def icon_clipboard():
    img = canvas(); d = ImageDraw.Draw(img)
    d.rounded_rectangle([50, 40, 206, 226], radius=14, outline=NEUTRAL, width=W)
    d.rounded_rectangle([94, 26, 162, 62], radius=10, outline=NEUTRAL, width=W)
    for y in (108, 144, 180):
        stroke(d, [(76, y), (180, y)], NEUTRAL, width=int(W*0.7))
    save(img, "clipboard-list")


# ── 24. pointer (cursor) ──────────────────────────────────────────────────────
def icon_pointer():
    img = canvas(); d = ImageDraw.Draw(img)
    stroke(d, [(90, 40), (90, 150)], NEUTRAL)
    stroke(d, [(130, 46), (130, 150)], NEUTRAL)
    stroke(d, [(170, 66), (170, 150)], NEUTRAL)
    stroke(d, [(210, 96), (210, 150)], NEUTRAL)
    stroke(d, [(90, 150), (60, 176), (76, 220), (190, 220), (210, 180), (210, 150)], NEUTRAL, closed=False)
    save(img, "pointer")


# ── 25. axe (firebreak) ───────────────────────────────────────────────────────
def icon_axe():
    img = canvas(); d = ImageDraw.Draw(img)
    stroke(d, [(112, 96), (46, 222)], PEACH, width=int(W*0.85))
    d.polygon([
        (112, 96), (108, 44), (140, 20), (188, 24),
        (206, 56), (188, 96), (140, 112),
    ], fill=PEACH)
    d.polygon([(140, 20), (188, 24), (176, 56), (132, 66)], fill=(30, 30, 46, 255))
    save(img, "axe")


# ── 26. check ─────────────────────────────────────────────────────────────────
def icon_check():
    img = canvas(); d = ImageDraw.Draw(img)
    stroke(d, [(40, 130), (100, 190), (216, 66)], GREEN, width=int(W*1.2))
    save(img, "check")


# ── 27. check-circle (apply) ──────────────────────────────────────────────────
def icon_check_circle():
    img = canvas(); d = ImageDraw.Draw(img)
    circle(d, 128, 128, 96, GREEN, width=W)
    stroke(d, [(80, 132), (114, 168), (180, 92)], GREEN, width=int(W*1.1))
    save(img, "check-circle")


# ── 28. plus ──────────────────────────────────────────────────────────────────
def icon_plus():
    img = canvas(); d = ImageDraw.Draw(img)
    stroke(d, [(128, 40), (128, 216)], BLUE, width=int(W*1.2))
    stroke(d, [(40, 128), (216, 128)], BLUE, width=int(W*1.2))
    save(img, "plus")


# ── 29. save ──────────────────────────────────────────────────────────────────
def icon_save():
    img = canvas(); d = ImageDraw.Draw(img)
    d.rounded_rectangle([44, 34, 212, 222], radius=12, outline=BLUE, width=W)
    d.rectangle([80, 34, 176, 92], fill=None, outline=BLUE, width=W)
    d.rectangle([76, 150, 180, 222], fill=None, outline=BLUE, width=W)
    save(img, "save")


# ── 30. folder-open ───────────────────────────────────────────────────────────
def icon_folder():
    img = canvas(); d = ImageDraw.Draw(img)
    stroke(d, [(30, 76), (30, 60), (46, 44), (96, 44), (116, 64), (206, 64), (206, 84)], YELLOW)
    stroke(d, [(30, 76), (216, 76), (188, 200), (54, 200), (30, 76)], YELLOW, closed=False)
    save(img, "folder-open")


# ── 31/32. corner-up-left / right (undo/redo) ────────────────────────────────
def icon_corner(right=False):
    img = canvas(); d = ImageDraw.Draw(img)
    if right:
        stroke(d, [(150, 86), (216, 140), (150, 194)], NEUTRAL)
        stroke(d, [(216, 140), (96, 140)], NEUTRAL)
        d.arc([16, 60, 176, 220], 180, 270, fill=NEUTRAL, width=W)
        save(img, "corner-up-right")
    else:
        stroke(d, [(106, 86), (40, 140), (106, 194)], NEUTRAL)
        stroke(d, [(40, 140), (160, 140)], NEUTRAL)
        d.arc([80, 60, 240, 220], 270, 360, fill=NEUTRAL, width=W)
        save(img, "corner-up-left")


# ── 33. trash-2 ───────────────────────────────────────────────────────────────
def icon_trash():
    img = canvas(); d = ImageDraw.Draw(img)
    stroke(d, [(40, 68), (216, 68)], PINK)
    d.rounded_rectangle([56, 68, 200, 224], radius=10, outline=PINK, width=W)
    stroke(d, [(96, 40), (96, 68)], PINK)
    stroke(d, [(160, 40), (160, 68)], PINK)
    stroke(d, [(96, 40), (160, 40)], PINK)
    stroke(d, [(100, 104), (100, 188)], PINK, width=int(W*0.7))
    stroke(d, [(156, 104), (156, 188)], PINK, width=int(W*0.7))
    save(img, "trash-2")


# ── 34. map-pin ───────────────────────────────────────────────────────────────
def icon_mappin():
    img = canvas(); d = ImageDraw.Draw(img)
    d.pieslice([48, 40, 208, 200], 200, 340, fill=PINK)
    d.polygon([(60, 150), (196, 150), (128, 236)], fill=PINK)
    dot(d, 128, 118, 30, (30, 30, 46, 255))
    save(img, "map-pin")


# ── 35/36. lock / unlock ─────────────────────────────────────────────────────
def icon_lock(locked=True):
    img = canvas(); d = ImageDraw.Draw(img)
    color = NEUTRAL if locked else DIM
    d.rounded_rectangle([50, 118, 206, 224], radius=14, outline=color, width=W)
    dot(d, 128, 172, 10, color)
    if locked:
        d.arc([80, 40, 176, 150], 180, 360, fill=color, width=W)
    else:
        d.arc([56, 40, 152, 150], 180, 360, fill=color, width=W)
    save(img, "lock" if locked else "unlock")


# ── 37. x (close) ─────────────────────────────────────────────────────────────
def icon_x():
    img = canvas(); d = ImageDraw.Draw(img)
    stroke(d, [(56, 56), (200, 200)], NEUTRAL, width=int(W*1.1))
    stroke(d, [(200, 56), (56, 200)], NEUTRAL, width=int(W*1.1))
    save(img, "x")


# ── 38. arrow-right ───────────────────────────────────────────────────────────
def icon_arrow_right():
    img = canvas(); d = ImageDraw.Draw(img)
    stroke(d, [(30, 128), (206, 128)], DIM)
    stroke(d, [(150, 74), (206, 128), (150, 182)], DIM)
    save(img, "arrow-right")


# ── 39. play ──────────────────────────────────────────────────────────────────
def icon_play():
    img = canvas(); d = ImageDraw.Draw(img)
    d.polygon([(70, 34), (70, 222), (216, 128)], fill=GREEN)
    save(img, "play")


# ── 40. sun (light-mode toggle) ───────────────────────────────────────────────
def icon_sun():
    img = canvas(); d = ImageDraw.Draw(img)
    circle(d, 128, 128, 52, YELLOW, fill=YELLOW)
    for ang in range(0, 360, 45):
        x1 = 128 + 72 * math.cos(math.radians(ang)); y1 = 128 + 72 * math.sin(math.radians(ang))
        x2 = 128 + 100 * math.cos(math.radians(ang)); y2 = 128 + 100 * math.sin(math.radians(ang))
        d.line([(x1, y1), (x2, y2)], fill=YELLOW, width=W)
    save(img, "sun")


# ── 41. moon (dark-mode toggle) ───────────────────────────────────────────────
def icon_moon():
    mask = Image.new("L", (S, S), 0)
    md = ImageDraw.Draw(mask)
    md.ellipse([40, 32, 216, 208], fill=255)     # main disc
    md.ellipse([92, 8, 248, 164], fill=0)         # offset cutout -> crescent
    img = canvas()
    ImageDraw.Draw(img).ellipse([40, 32, 216, 208], fill=BLUE)
    img.putalpha(mask)
    save(img, "moon")


# ── New icons: drone and route/trajectory ─────────────────────────────────────────────

def icon_drone():
    """Quadcopter viewed from above: cross body + 4 rotors + camera dot."""
    img = canvas()
    d = ImageDraw.Draw(img)
    cx, cy = S // 2, S // 2
    arm = 72
    rotor_r = 48
    body_r  = 28
    w2 = 10
    ac = TEAL
    for angle in (45, 135, 225, 315):
        rad = math.radians(angle)
        ex = cx + arm * math.cos(rad)
        ey = cy + arm * math.sin(rad)
        stroke(d, [(cx, cy), (ex, ey)], ac, width=w2)
        circle(d, ex, ey, rotor_r, ac, width=9)
    dot(d, cx, cy, body_r, ac)
    dot(d, cx, cy, 12, (30, 30, 46, 255))
    save(img, "drone")


def icon_route():
    """Drone flight path: dashed line + waypoint dots + arrowhead."""
    img = canvas()
    d = ImageDraw.Draw(img)
    rc = BLUE
    dot_r = 14
    pts = [(40, 200), (80, 160), (128, 128), (176, 96), (216, 60)]
    dash_len, gap_len = 18, 12
    drawing, budget = True, 0.0
    for i in range(len(pts) - 1):
        x0, y0 = pts[i]
        x1, y1 = pts[i + 1]
        seg_len = math.hypot(x1 - x0, y1 - y0)
        dist = 0.0
        while dist < seg_len:
            if budget <= 0:
                drawing = not drawing
                budget = dash_len if drawing else gap_len
            step = min(seg_len - dist, budget)
            frac_s = dist / seg_len
            frac_e = (dist + step) / seg_len
            if drawing:
                stroke(d,
                       [(x0 + frac_s*(x1-x0), y0 + frac_s*(y1-y0)),
                        (x0 + frac_e*(x1-x0), y0 + frac_e*(y1-y0))],
                       rc, width=14)
            dist += step
            budget -= step
    last, prev = pts[-1], pts[-2]
    angle = math.atan2(last[1]-prev[1], last[0]-prev[0])
    hs = 32
    tip = last
    bl = (tip[0]-hs*math.cos(angle)+hs*0.45*math.sin(angle),
          tip[1]-hs*math.sin(angle)-hs*0.45*math.cos(angle))
    br = (tip[0]-hs*math.cos(angle)-hs*0.45*math.sin(angle),
          tip[1]-hs*math.sin(angle)+hs*0.45*math.cos(angle))
    d.polygon([tip, bl, br], fill=rc)
    for p in pts[:-1]:
        dot(d, p[0], p[1], dot_r, rc)
        dot(d, p[0], p[1], dot_r//2, (30, 30, 46, 200))
    save(img, "route")


if __name__ == "__main__":
    icon_flame()
    icon_ash()
    icon_brain()
    icon_flask()
    icon_barchart()
    icon_target()
    icon_alert()
    icon_box()
    icon_wind()
    icon_droplet()
    icon_volume(muted=False)
    icon_volume(muted=True)
    icon_wrench()
    icon_bug()
    icon_truck()
    icon_satellite()
    icon_leaf()
    icon_arrow_up()
    icon_thermometer()
    icon_zap()
    icon_clock()
    icon_cloud_sun()
    icon_clipboard()
    icon_pointer()
    icon_axe()
    icon_check()
    icon_check_circle()
    icon_plus()
    icon_save()
    icon_folder()
    icon_corner(right=False)
    icon_corner(right=True)
    icon_trash()
    icon_mappin()
    icon_lock(locked=True)
    icon_lock(locked=False)
    icon_x()
    icon_arrow_right()
    icon_play()
    icon_sun()
    icon_moon()
    icon_drone()
    icon_route()
    print("ALL ICONS GENERATED")


# ── New icons: drone and route/trajectory ─────────────────────────────────────

def icon_drone():
    """Quadcopter viewed from above: cross body + 4 rotors + camera dot."""
    img = canvas()
    d = ImageDraw.Draw(img)
    cx, cy = S // 2, S // 2
    arm = 72          # arm half-length from centre
    rotor_r = 48      # rotor circle radius
    body_r  = 28      # central hub radius
    w2 = 10           # thin arm stroke

    # Arm colour: sky-blue TEAL
    ac = TEAL
    # 4 arms at 45°
    for angle in (45, 135, 225, 315):
        rad = math.radians(angle)
        ex = cx + arm * math.cos(rad)
        ey = cy + arm * math.sin(rad)
        stroke(d, [(cx, cy), (ex, ey)], ac, width=w2)
        # Rotor circle
        circle(d, ex, ey, rotor_r, ac, width=9)

    # Central body hub (filled circle)
    dot(d, cx, cy, body_r, ac)
    # Camera dot (lens) — tiny dark circle inside hub
    dot(d, cx, cy, 12, (30, 30, 46, 255))

    save(img, "drone")


def icon_route():
    """Drone flight path: curved dashed line with waypoint dots + arrowhead."""
    img = canvas()
    d = ImageDraw.Draw(img)
    # Route colour: BLUE
    rc = BLUE
    dot_r = 14

    # Control points for a gentle S-curve (Bézier approximated as polyline)
    pts = [
        (40,  200),
        (80,  160),
        (128, 128),
        (176,  96),
        (216,  60),
    ]

    # Draw dashed polyline: alternate filled/gap segments
    dash_len = 18
    gap_len  = 12
    segment_pts = []
    t = 0.0
    drawing = True
    budget = 0.0

    for i in range(len(pts) - 1):
        x0, y0 = pts[i]
        x1, y1 = pts[i + 1]
        seg_len = math.hypot(x1 - x0, y1 - y0)
        dist = 0.0
        while dist < seg_len:
            if budget <= 0:
                drawing = not drawing
                budget = dash_len if drawing else gap_len
            step = min(seg_len - dist, budget)
            frac_s = dist / seg_len
            frac_e = (dist + step) / seg_len
            sx = x0 + frac_s * (x1 - x0)
            sy = y0 + frac_s * (y1 - y0)
            ex = x0 + frac_e * (x1 - x0)
            ey = y0 + frac_e * (y1 - y0)
            if drawing:
                stroke(d, [(sx, sy), (ex, ey)], rc, width=14)
            dist += step
            budget -= step

    # Arrowhead at the last point
    last = pts[-1]
    prev = pts[-2]
    angle = math.atan2(last[1] - prev[1], last[0] - prev[0])
    hs = 32   # arrowhead half-size
    tip = last
    bl = (tip[0] - hs * math.cos(angle) + hs * 0.45 * math.sin(angle),
          tip[1] - hs * math.sin(angle) - hs * 0.45 * math.cos(angle))
    br = (tip[0] - hs * math.cos(angle) - hs * 0.45 * math.sin(angle),
          tip[1] - hs * math.sin(angle) + hs * 0.45 * math.cos(angle))
    d.polygon([tip, bl, br], fill=rc)

    # Waypoint dots along the route
    for p in pts[:-1]:
        dot(d, p[0], p[1], dot_r, rc)
        dot(d, p[0], p[1], dot_r // 2, (30, 30, 46, 200))

    save(img, "route")

