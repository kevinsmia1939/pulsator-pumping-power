import cv2
import numpy as np
import argparse
import csv
from pathlib import Path
import sys
import math
import time

# =========================
# Args
# =========================
def parse_args():
    p = argparse.ArgumentParser(
        description="Track moving OR brightest blob and read a 7-seg timer (with optional perspective correction). CSV has ROI header then time_LED,x,y."
    )
    p.add_argument("video", help="Path to video or camera index (e.g., 0).")

    # Performance / capture
    p.add_argument("--threads", type=int, default=None, help="Set OpenCV thread count.")
    p.add_argument("--progress", type=int, default=0, help="Print progress every N frames (0=silent).")
    p.add_argument("--backend", choices=["any","dshow","v4l2","gstreamer","ffmpeg"], default="any")
    p.add_argument("--cam-warmup", type=int, default=30)
    p.add_argument("--first-frame-tries", type=int, default=120)
    p.add_argument("--first-frame-brightness", type=float, default=8.0)

    # Rotation (applied before any processing/ROI)
    p.add_argument("--rotate", choices=["none","cw","ccw","180"], default="none",
                   help="Rotate every frame before processing: cw=90° clockwise, ccw=90° counter-clockwise.")

    # Main ROI (water level)
    p.add_argument("--roi", type=int, nargs=4, metavar=("X","Y","W","H"),
                   help="Main ROI (x y w h) in the ROTATED frame. If omitted and no selection, full frame is used.")
    p.add_argument("--select-roi", action="store_true",
                   help="Interactively select MAIN ROI on a chosen (rotated) frame.")
    p.add_argument("--roi-select-scale", type=float, default=1.0,
                   help="Extra scale factor for MAIN ROI selection window only (multiplies preview scale).")

    # Use any frame for selections
    p.add_argument("--select-at-frame", type=int, default=-1,
                   help="Use this 0-based frame index for --select-roi/--select-digits-quad/--select-digits-roi.")
    p.add_argument("--select-at-seconds", type=float, default=None,
                   help="Use this timestamp (sec) for --select-roi/--select-digits-quad/--select-digits-roi.")

    # Tracking mode
    p.add_argument("--track-mode", choices=["motion","brightest"], default="motion",
                   help="motion: absdiff vs background. brightest: largest/brightest blob after filtering.")
    p.add_argument("--bright-thresh", type=int, default=0,
                   help="Brightest mode: if >0, fixed threshold on gray (0..255). If 0, Otsu (after blur).")

    # 7-seg timer ROI (rectangular crop INSIDE warped plane)
    p.add_argument("--digits-roi", type=int, nargs=4, metavar=("X","Y","W","H"),
                   help="Rectangular digits ROI (x y w h) in the WARPED plane (if a quad is used) or in the ROTATED frame otherwise.")
    p.add_argument("--select-digits-roi", action="store_true",
                   help="Interactively select digits ROI. If a quad is used, this selection happens on the WARPED preview.")

    # 7-seg timer perspective correction
    p.add_argument("--digits-quad", type=float, nargs=8, metavar=("x1","y1","x2","y2","x3","y3","x4","y4"),
                   help="Four points (in ROTATED frame) around the display. Any order; auto-ordered TL,TR,BR,BL.")
    p.add_argument("--select-digits-quad", action="store_true",
                   help="Interactively click 4 corners of the digits display (in ROTATED frame).")
    p.add_argument("--digits-warp-size", type=int, nargs=2, metavar=("W","H"),
                   help="Output size (width height) for the perspective warp. If omitted, size is auto-estimated.")

    # 7-seg recognition parameters
    p.add_argument("--digits", type=int, default=0,
                   help="Expected number of digits (e.g., 3). 0=auto split by projection.")
    p.add_argument("--digits-allow-fewer", action="store_true",
                   help="When --digits > 0, fixed equal slices; drop only leading blanks (e.g., __3 -> 3).")
    p.add_argument("--seg-thresh", type=float, default=0.40,
                   help="Segment ON threshold (fraction of lit pixels in each segment region).")
    p.add_argument("--seg-off-thresh", type=float, default=None,
                   help="Segment OFF threshold (fraction). If not set, uses (seg-thresh - seg-hyst).")
    p.add_argument("--seg-hyst", type=float, default=0.10,
                   help="Hysteresis gap: OFF threshold = seg-thresh - seg-hyst (if --seg-off-thresh not set).")
    p.add_argument("--digit-split-thresh", type=float, default=0.10,
                   help="Auto-split: column-sum threshold as fraction of max to separate digits (for --digits=0).")
    p.add_argument("--digit-margin", type=float, default=0.10,
                   help="When digits>0, crop margin on left/right of each equal slice (fraction of slice width).")
    p.add_argument("--digits-color", choices=["auto","red","green","blue","mono"], default="auto",
                   help="Color gating inside digits ROI. 'auto' assumes red, 'mono' uses grayscale/otsu.")
    p.add_argument("--digits-s-min", type=int, default=120, help="HSV S min for digits ROI color gate (0..255).")
    p.add_argument("--digits-v-min", type=int, default=80,  help="HSV V min for digits ROI color gate (0..255).")
    p.add_argument("--digits-delta", type=int, default=20,
                   help="Channel-dominance for color LEDs (e.g., red: R>G+Δ & R>B+Δ). 0=off.")
    p.add_argument("--digits-min-intensity", type=int, default=0,
                   help="Digits ROI: ignore pixels darker than this (0..255). Bright polarity only.")
    p.add_argument("--digits-polarity", choices=["bright","dark"], default="bright",
                   help="If your display has dark segments on bright background, choose 'dark'.")
    p.add_argument("--digits-open", type=int, default=1, help="Morph open radius on 7-seg mask (0..3).")
    p.add_argument("--digits-close", type=int, default=1, help="Morph close radius on 7-seg mask (0..3).")
    p.add_argument("--digits-debounce", type=int, default=2,
                   help="Frames the same value must persist before accepting an update (helps with flicker).")
    p.add_argument("--enforce-upcounter", action="store_true",
                   help="Only accept new 7-seg value if same as last or exactly +1.")

    p.add_argument("--show-digits", action="store_true",
                   help="Show a window with the 7-seg binary mask and per-digit boxes (for tuning).")
    p.add_argument("--show-seg-boxes", action="store_true",
                   help="Overlay the a..g sampling rectangles (green=ON, red=OFF) in the 7-seg debug window.")

    # Background (MAIN ROI)
    p.add_argument("--bg-method", choices=["median","running"], default="median",
                   help="Background: median of warmup frames or running average.")
    p.add_argument("--warmup", type=int, default=30, help="Frames for median background (or init).")
    p.add_argument("--running-alpha", type=float, default=0.01, help="Running average learning rate (0..1).")

    # Motion/brightness segmentation (MAIN ROI)
    p.add_argument("--blur", type=int, default=3, help="Gaussian blur kernel (odd). 0/1 = off.")
    p.add_argument("--thresh", type=int, default=0, help="Binary threshold on absdiff; 0 = Otsu.")
    p.add_argument("--open", type=int, default=1, help="Morph open radius (0..3).")
    p.add_argument("--close", type=int, default=1, help="Morph close radius (0..3).")

    # Brightness gate (MAIN ROI)
    p.add_argument("--min-intensity", type=int, default=0, help="Ignore pixels darker than this (0..255).")

    # Color preference (MAIN ROI, HSV)
    p.add_argument("--prefer-color", choices=["none","blue","green","red"], default="none",
                   help="Prefer a color by gating & scoring (HSV). Default: none.")
    p.add_argument("--color-s-min", type=int, default=80, help="Min saturation for color gate (0..255).")
    p.add_argument("--color-v-min", type=int, default=60, help="Min value/brightness for color gate (0..255).")
    p.add_argument("--color-weight", type=float, default=0.75,
                   help="Extra weight for preferred color pixels in scoring (0=no preference).")

    # Output / preview
    p.add_argument("--display", action="store_true", help="Show main annotated window.")
    p.add_argument("--resize", type=int, nargs=2, metavar=("W","H"),
                   help="Preview resize in pixels (overrides --resize-scale and --display-scale).")
    p.add_argument("--resize-scale", type=float, default=None,
                   help="Uniform preview scale factor. Example: 0.5 halves size, 1.5 enlarges. Overrides --display-scale. Also applies to selection previews.")
    p.add_argument("--display-scale", type=float, default=1.0,
                   help="Uniform scale factor for preview windows (overlay, filter, 7seg debug).")
    p.add_argument("--show-filter", action="store_true",
                   help="Show separate window preview of the actual mask used (within MAIN ROI).")
    p.add_argument("--csv", type=Path, default=None,
                   help="CSV output path. If omitted, will use <video_basename>-track.csv in the same folder.")
    p.add_argument("--out", type=Path, help="Write annotated MP4.")

    # Timing (for perf stats only)
    p.add_argument("--fps", type=float, default=0.0, help="FPS only for log display if you want.")

    # ===== Cosine correction =====
    p.add_argument("--cam-distance", type=float, default=0.0,
                   help="Camera distance to motion axis (meters). If 0 or omitted, cosine correction is disabled.")
    p.add_argument("--cam-offset", type=float, default=0.0,
                   help="Lateral offset between camera optical axis and motion axis (meters). Used with --cam-distance.")

    return p.parse_args()

# =========================
# Utils
# =========================
def backend_flag(name: str):
    return {"any":0,"dshow":cv2.CAP_DSHOW,"v4l2":cv2.CAP_V4L2,"gstreamer":cv2.CAP_GSTREAMER,"ffmpeg":cv2.CAP_FFMPEG}.get(name,0)

def open_capture(source, backend_name):
    flag = backend_flag(backend_name)
    try:
        cam_idx = int(source)
        return cv2.VideoCapture(cam_idx, flag) if flag else cv2.VideoCapture(cam_idx)
    except ValueError:
        return cv2.VideoCapture(source, flag) if flag else cv2.VideoCapture(source)

def is_nonblack(frame, thresh_mean=8.0):
    if frame is None or frame.size == 0:
        return False
    g = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return float(g.mean()) > float(thresh_mean)

def rotate_frame(img, mode: str):
    if mode == "cw":  return cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
    if mode == "ccw": return cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
    if mode == "180": return cv2.rotate(img, cv2.ROTATE_180)
    return img

def grab_first_visible_frame(cap, tries=120, brightness_thresh=8.0, cam_warmup=0):
    for _ in range(max(0, cam_warmup)):
        if not cap.grab():
            break
    ok_ret, frame = False, None
    for _ in range(max(1, tries)):
        ok, f = cap.read()
        if not ok:
            break
        frame = f
        if is_nonblack(f, brightness_thresh):
            ok_ret = True
            break
    return ok_ret, frame

def clamp_roi(frame, roi):
    x,y,w,h = roi
    H,W = frame.shape[:2]
    x = max(0, min(x, W-1)); y = max(0, min(y, H-1))
    w = max(1, min(w, W-x)); h = max(1, min(h, H-y))
    return (x,y,w,h)

def pick_roi_interactive(frame, title="Select ROI (ENTER/SPACE=OK, C=Cancel)", scale=1.0):
    s = float(scale) if (scale and scale > 0) else 1.0
    if abs(s - 1.0) > 1e-3:
        win_title = f"{title} [scaled {s:.2f}x]"
        disp = cv2.resize(
            frame, None, fx=s, fy=s,
            interpolation=cv2.INTER_AREA if s < 1.0 else cv2.INTER_LINEAR
        )
        r = cv2.selectROI(win_title, disp, fromCenter=False, showCrosshair=False)
        cv2.destroyWindow(win_title)
        x, y, w, h = map(int, r)
        if w > 0 and h > 0:
            x = int(round(x / s)); y = int(round(y / s))
            w = int(round(w / s)); h = int(round(h / s))
            return (x, y, w, h)
        return None
    else:
        r = cv2.selectROI(title, frame, fromCenter=False, showCrosshair=False)
        cv2.destroyWindow(title)
        x, y, w, h = map(int, r)
        return (x, y, w, h) if w > 0 and h > 0 else None

def to_gray(bgr):
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)

# ---- Selection frame helpers ----
def read_frame_at(cap, target_idx, allow_step=True):
    """Try to seek to target_idx; if not supported, step forward."""
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    if total > 0 and 0 <= target_idx < total:
        if cap.set(cv2.CAP_PROP_POS_FRAMES, target_idx):
            ok, f = cap.read()
            if ok:
                return True, f, target_idx
    # fallback: step from start
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    idx = 0
    while idx < target_idx:
        ok = cap.grab()
        if not ok:
            break
        idx += 1
    ok, f = cap.read()
    if ok:
        return True, f, idx
    return False, None, -1

# =========================
# Perspective helpers (digits quad)
# =========================
def order_quad(pts):
    pts = np.array(pts, dtype=np.float32)
    s = pts.sum(axis=1)
    diff = np.diff(pts, axis=1).ravel()
    tl = pts[np.argmin(s)]
    br = pts[np.argmax(s)]
    tr = pts[np.argmin(diff)]
    bl = pts[np.argmax(diff)]
    return np.array([tl, tr, br, bl], dtype=np.float32)

def estimate_warp_size(quad):
    (tl, tr, br, bl) = quad
    wA = np.linalg.norm(br - bl)
    wB = np.linalg.norm(tr - tl)
    hA = np.linalg.norm(tr - br)
    hB = np.linalg.norm(tl - bl)
    W = int(round(max(wA, wB)))
    H = int(round(max(hA, hB)))
    return (max(20, W), max(20, H))

def compute_perspective(quad, out_size=None):
    quad = order_quad(quad)
    if out_size is None:
        out_size = estimate_warp_size(quad)
    W, H = out_size
    dst = np.array([[0,0],[W-1,0],[W-1,H-1],[0,H-1]], dtype=np.float32)
    M = cv2.getPerspectiveTransform(quad, dst)
    return M, (W, H), quad

def pick_quad_interactive(img, title="Click 4 corners (ENTER=OK, R=reset, C=cancel)"):
    pts = []
    window = title

    def cb(event, x, y, flags, param):
        nonlocal pts
        if event == cv2.EVENT_LBUTTONDOWN and len(pts) < 4:
            pts.append((x, y))
        elif event == cv2.EVENT_RBUTTONDOWN and pts:
            pts.pop()

    cv2.namedWindow(window)
    cv2.setMouseCallback(window, cb)

    while True:
        vis = img.copy()
        for i, (x, y) in enumerate(pts):
            cv2.circle(vis, (x, y), 4, (0, 255, 255), -1)
            cv2.putText(vis, f"{i+1}", (x+6, y-6), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,0), 2, cv2.LINE_AA)
            cv2.putText(vis, f"{i+1}", (x+6, y-6), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 1, cv2.LINE_AA)
        if len(pts) >= 2:
            cv2.polylines(vis, [np.array(pts, dtype=np.int32)], isClosed=False, color=(0, 255, 255), thickness=2)
        cv2.putText(vis, "Click 4 corners of the digits display.", (8, 24),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,0), 3, cv2.LINE_AA)
        cv2.putText(vis, "Click 4 corners of the digits display.", (8, 24),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 1, cv2.LINE_AA)
        cv2.putText(vis, "ENTER=OK  R=reset  C=cancel  Right-click=undo", (8, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,0), 3, cv2.LINE_AA)
        cv2.putText(vis, "ENTER=OK  R=reset  C=cancel  Right-click=undo", (8, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200,255,200), 1, cv2.LINE_AA)
        cv2.imshow(window, vis)
        k = cv2.waitKey(10) & 0xFF
        if k in (13, 10, 32) and len(pts) == 4:
            cv2.destroyWindow(window)
            return order_quad(pts)
        elif k in (ord('r'), ord('R')):
            pts = []
        elif k in (ord('c'), ord('C'), 27):
            cv2.destroyWindow(window)
            return None

# =========================
# Background (MAIN ROI)
# =========================
def build_bg_median(cap, roi, warmup, rotate_mode="none"):
    frames = []
    pos0 = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
    for _ in range(max(1, warmup)):
        ok, f = cap.read()
        if not ok: break
        f = rotate_frame(f, rotate_mode)
        if roi:
            x, y, w, h = roi
            f = f[y:y+h, x:x+w]
        frames.append(to_gray(f))
    if not frames:
        raise SystemExit("No frames gathered for median background.")
    bg = np.median(np.stack(frames, 0), 0).astype(np.uint8)
    cap.set(cv2.CAP_PROP_POS_FRAMES, pos0)
    return bg.astype(np.float32)

def init_bg_running(gray):
    return gray.astype(np.float32)

def update_bg_running(bg_f32, gray, alpha):
    cv2.accumulateWeighted(gray, bg_f32, alpha)
    return bg_f32

# =========================
# Motion / Brightest masks
# =========================
def motion_mask(gray, bg_uint8, blur_k=3, thr=0, r_open=1, r_close=1):
    gray_proc = gray
    bg_proc = bg_uint8
    if blur_k and blur_k % 2 == 1 and blur_k > 1:
        gray_proc = cv2.GaussianBlur(gray, (blur_k, blur_k), 0)
        bg_proc   = cv2.GaussianBlur(bg_uint8, (blur_k, blur_k), 0)
    diff = cv2.absdiff(gray_proc, bg_proc)
    if thr <= 0:
        _, mask = cv2.threshold(diff, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    else:
        _, mask = cv2.threshold(diff, thr, 255, cv2.THRESH_BINARY)
    if r_open > 0:
        k = 1 + 2*r_open
        ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k,k))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, ker, 1)
    if r_close > 0:
        k = 1 + 2*r_close
        ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k,k))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, ker, 1)
    return mask, diff, gray_proc

def brightest_mask(gray_proc, min_intensity=0, bright_thresh=0, r_open=1, r_close=1):
    if bright_thresh > 0:
        _, thr = cv2.threshold(gray_proc, int(bright_thresh), 255, cv2.THRESH_BINARY)
    else:
        _, thr = cv2.threshold(gray_proc, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    if min_intensity > 0:
        gate = (gray_proc >= np.uint8(min_intensity)).astype(np.uint8) * 255
        m = cv2.bitwise_and(thr, gate)
    else:
        m = thr

    if r_open > 0:
        k = 1 + 2*r_open
        ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k,k))
        m = cv2.morphologyEx(m, cv2.MORPH_OPEN, ker, 1)
    if r_close > 0:
        k = 1 + 2*r_close
        ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k,k))
        m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, ker, 1)

    return m

# =========================
# Color helpers (HSV) for MAIN ROI preference
# =========================
def color_mask_hsv(bgr, which: str, s_min: int, v_min: int):
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    s_min_u8 = np.uint8(np.clip(s_min, 0, 255))
    v_min_u8 = np.uint8(np.clip(v_min, 0, 255))
    if which == "blue":
        return cv2.inRange(hsv, np.array([90,  s_min_u8, v_min_u8]), np.array([140, 255, 255]))
    if which == "green":
        return cv2.inRange(hsv, np.array([35,  s_min_u8, v_min_u8]), np.array([85,  255, 255]))
    if which == "red":
        m1 = cv2.inRange(hsv, np.array([0,   s_min_u8, v_min_u8]), np.array([10,  255, 255]))
        m2 = cv2.inRange(hsv, np.array([160, s_min_u8, v_min_u8]), np.array([179, 255, 255]))
        return cv2.bitwise_or(m1, m2)
    return None

# =========================
# 7-seg decoder with hysteresis
# =========================
SEG_PATTERNS = {0:0x3F,1:0x06,2:0x5B,3:0x4F,4:0x66,5:0x6D,6:0x7D,7:0x07,8:0x7F,9:0x6F}
SEG_LABELS   = {0:'a',1:'b',2:'c',3:'d',4:'e',5:'f',6:'g'}

def _dominance_mask(bgr, color: str, delta: int):
    if delta <= 0: return None
    b, g, r = cv2.split(bgr)
    b = b.astype(np.int16); g = g.astype(np.int16); r = r.astype(np.int16)
    if color == "red":
        return (((r-g) > delta) & ((r-b) > delta)).astype(np.uint8) * 255
    if color == "green":
        return (((g-r) > delta) & ((g-b) > delta)).astype(np.uint8) * 255
    if color == "blue":
        return (((b-r) > delta) & ((b-g) > delta)).astype(np.uint8) * 255
    return None

def _morph(mask, r_open: int, r_close: int):
    if r_open > 0:
        k = 1 + 2*r_open
        ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k,k))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, ker, 1)
    if r_close > 0:
        k = 1 + 2*r_close
        ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k,k))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, ker, 1)
    return mask

def hsv_mask_for_digits(bgr, mode, s_min, v_min, delta, polarity, r_open, r_close, min_intensity=0):
    if mode == "auto":
        mode = "red"

    if mode == "mono":
        g = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        _, m = cv2.threshold(g, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    else:
        hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
        s_min = int(np.clip(s_min, 0, 255)); v_min = int(np.clip(v_min, 0, 255))
        if mode == "red":
            m1 = cv2.inRange(hsv, np.array([0,   s_min, v_min]), np.array([10,  255, 255]))
            m2 = cv2.inRange(hsv, np.array([160, s_min, v_min]), np.array([179, 255, 255]))
            m  = cv2.bitwise_or(m1, m2)
        elif mode == "blue":
            m = cv2.inRange(hsv, np.array([90, s_min, v_min]), np.array([140, 255, 255]))
        elif mode == "green":
            m = cv2.inRange(hsv, np.array([35, s_min, v_min]), np.array([85, 255, 255]))
        else:
            g = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
            _, m = cv2.threshold(g, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        dom = _dominance_mask(bgr, mode, delta)
        if dom is not None:
            m = cv2.bitwise_and(m, dom)

    if min_intensity > 0 and polarity == "bright":
        g = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        gate = (g >= np.uint8(min_intensity)).astype(np.uint8) * 255
        m = cv2.bitwise_and(m, gate)

    if polarity == "dark":
        m = cv2.bitwise_not(m)

    m = _morph(m, r_open, r_close)
    return m

def split_digits(mask, expected_n=0, split_thresh_frac=0.10, margin_frac=0.10):
    H, W = mask.shape[:2]
    boxes = []
    if expected_n and expected_n > 0:
        slice_w = W / float(expected_n)
        for i in range(expected_n):
            x0 = int(round(i * slice_w)); x1 = int(round((i + 1) * slice_w))
            w = max(1, x1 - x0)
            l = int(round(w * margin_frac))
            x = x0 + l
            w2 = max(1, w - 2 * l)
            boxes.append((x, 0, w2, H))
        return boxes
    colsum = mask.sum(axis=0).astype(np.float32)
    if colsum.max() <= 0: return []
    th = float(colsum.max()) * float(split_thresh_frac)
    active = (colsum > th).astype(np.uint8)
    inside = False; run_start = 0
    for x in range(W):
        if not inside and active[x]:
            inside = True; run_start = x
        elif inside and not active:
            inside = False; boxes.append((run_start, 0, x - run_start, H))
    if inside:
        boxes.append((run_start, 0, W - run_start, H))
    boxes = [b for b in boxes if b[2] > max(3, W//50)]
    return boxes

def get_seg_boxes(W, H):
    xMl = int(W * 0.25); xMr = int(W * 0.75)
    yT = int(H * 0.05); yB = int(H * 0.95)
    yU = int(H * 0.20); yM = int(H * 0.50); yD = int(H * 0.80)
    return {
        0: (xMl, max(0, yT), max(1, xMr - xMl), max(1, int(H*0.12))),                 # a
        1: (xMr, yU, max(1, W - xMr), max(1, yM - yU)),                                # b
        2: (xMr, yM, max(1, W - xMr), max(1, yD - yM)),                                # c
        3: (xMl, min(yB - int(H*0.12), yD), max(1, xMr - xMl), max(1, int(H*0.12))),  # d
        4: (0,   yM, max(1, xMl), max(1, yD - yM)),                                    # e
        5: (0,   yU, max(1, xMl), max(1, yM - yU)),                                    # f
        6: (xMl, max(0, int(yM - H*0.06)), max(1, xMr - xMl), max(1, int(H*0.12))),    # g
    }

def segments_from_digit(mask_digit, on_thresh=0.40, off_thresh=None, prev_pattern=None):
    H, W = mask_digit.shape[:2]
    seg_boxes = get_seg_boxes(W, H)

    if off_thresh is None:
        off_thresh = max(0.0, min(on_thresh, on_thresh - 0.10))

    pattern = 0
    for seg_id, (x, y, w, h) in seg_boxes.items():
        x = np.clip(x, 0, max(0, W-1)); y = np.clip(y, 0, max(0, H-1))
        w = max(1, min(w, W - x)); h = max(1, min(h, H - y))
        roi = mask_digit[y:y+h, x:x+w]
        lit_frac = float((roi > 0).sum()) / float(w*h)

        prev_bit = 0
        if prev_pattern is not None:
            prev_bit = 1 if (prev_pattern & (1 << seg_id)) else 0

        if lit_frac >= on_thresh:
            bit = 1
        elif lit_frac <= off_thresh:
            bit = 0
        else:
            bit = prev_bit

        if bit:
            pattern |= (1 << seg_id)

    return pattern

def pattern_to_digit(pattern, tolerate_hamming=1):
    if pattern == 0:
        return None
    for d, p in SEG_PATTERNS.items():
        if p == pattern:
            return d
    best_d, best_hd = None, 10
    for d, p in SEG_PATTERNS.items():
        hd = bin(p ^ pattern).count("1")
        if hd < best_hd:
            best_d, best_hd = d, hd
    return best_d if best_hd <= int(tolerate_hamming) else None

def read_digits_from_roi(bgr, expected_n=0, mode="auto", s_min=120, v_min=80,
                         delta=20, polarity="bright", seg_on_thresh=0.40,
                         split_thresh_frac=0.10, margin_frac=0.10,
                         r_open=1, r_close=1, min_intensity=0,
                         allow_fewer=False,
                         prev_patterns=None, seg_off_thresh=None, seg_hyst=0.10):
    mask = hsv_mask_for_digits(bgr, mode, s_min, v_min, delta, polarity, r_open, r_close, min_intensity=min_intensity)

    if expected_n and expected_n > 0:
        boxes = split_digits(mask, expected_n=expected_n,
                             split_thresh_frac=split_thresh_frac,
                             margin_frac=margin_frac)
        if not boxes:
            return None, mask, [], prev_patterns
    else:
        boxes = split_digits(mask, expected_n=0,
                             split_thresh_frac=split_thresh_frac,
                             margin_frac=margin_frac)
        if not boxes:
            return None, mask, [], prev_patterns

    if seg_off_thresh is None:
        seg_off_thresh = float(seg_on_thresh) - float(seg_hyst)
    seg_off_thresh = max(0.0, min(seg_off_thresh, seg_on_thresh))

    if prev_patterns is None or len(prev_patterns) != len(boxes):
        prev_patterns = [0] * len(boxes)

    patterns_now = []
    decoded = []
    for i, (x, y, w, h) in enumerate(boxes):
        dmask = mask[y:y+h, x:x+w]
        pat = segments_from_digit(
            dmask, on_thresh=seg_on_thresh,
            off_thresh=seg_off_thresh,
            prev_pattern=prev_patterns[i]
        )
        patterns_now.append(pat)

        if expected_n and expected_n > 0:
            if pat == 0:
                decoded.append(None)
                continue
            d = pattern_to_digit(pat, tolerate_hamming=1)
            if d is None:
                return None, mask, boxes, patterns_now
            decoded.append(d)
        else:
            d = pattern_to_digit(pat, tolerate_hamming=1)
            if d is None:
                return None, mask, boxes, patterns_now
            decoded.append(d)

    if expected_n and expected_n > 0:
        if allow_fewer:
            i = 0
            while i < len(decoded) and decoded[i] is None:
                i += 1
            if i == len(decoded):
                return None, mask, boxes, patterns_now
            tail = decoded[i:]
            if any(d is None for d in tail):
                return None, mask, boxes, patterns_now
            digits = tail
        else:
            if any(d is None for d in decoded):
                return None, mask, boxes, patterns_now
            digits = decoded
    else:
        digits = decoded

    value = 0
    for d in digits:
        value = value * 10 + int(d)
    return value, mask, boxes, patterns_now

def render_digits_debug(mask, boxes, value, seg_thresh=0.40, show_seg_boxes=False):
    vis = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    for (x, y, w, h) in boxes:
        cv2.rectangle(vis, (x, y), (x+w, y+h), (0, 255, 0), 1)
        if show_seg_boxes:
            seg_boxes = get_seg_boxes(w, h)
            for seg_id, (sx, sy, sw, sh) in seg_boxes.items():
                gx0, gy0 = x + sx, y + sy
                gx1, gy1 = gx0 + sw, gy0 + sh
                roi = mask[gy0:gy1, gx0:gx1]
                area = max(1, sw * sh)
                frac = float((roi > 0).sum()) / float(area)
                on = frac >= seg_thresh
                color = (0, 200, 0) if on else (0, 0, 255)
                cv2.rectangle(vis, (gx0, gy0), (gx1, gy1), color, 1)
                cv2.putText(vis, SEG_LABELS[seg_id], (gx0+1, gy0+12),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1, cv2.LINE_AA)
    if value is not None:
        cv2.putText(vis, str(value), (5, 16),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,255), 1, cv2.LINE_AA)
    return vis

# =========================
# Component picker
# =========================
def pick_brightest_component(mask_bin, diff_or_gray, bright_mask=None, weight=None):
    if bright_mask is not None:
        mask_bin = cv2.bitwise_and(mask_bin, bright_mask)
    num, labels, stats, _ = cv2.connectedComponentsWithStats(mask_bin, connectivity=8)
    if num <= 1:
        return None, 0.0
    best_idx, best_score = -1, -1.0
    use = weight.astype(np.float32) if weight is not None else diff_or_gray.astype(np.float32)
    for i in range(1, num):
        x, y, w, h, area = stats[i]
        if area <= 0: continue
        roi_labels = labels[y:y+h, x:x+w]
        roi_use = use[y:y+h, x:x+w]
        score = float(roi_use[roi_labels == i].sum())
        if score > best_score:
            best_score, best_idx = score, i
    if best_idx < 0:
        return None, 0.0
    x, y, w, h, _ = stats[best_idx]
    return (int(x), int(y), int(w), int(h)), best_score

# =========================
# Main
# =========================
def main():
    args = parse_args()

    if args.threads is not None and args.threads >= 0:
        try: cv2.setNumThreads(args.threads)
        except Exception: pass
    try: cv2.setUseOptimized(True)
    except Exception: pass

    # --- preview scale helpers ---
    def preview_scale_factor():
        """Scale factor for preview windows when not using --resize WxH."""
        return args.resize_scale if (args.resize_scale is not None) else float(args.display_scale)

    def apply_preview_resize(img):
        """Applies preview sizing for display/debug windows (not used for ROI selection frames)."""
        if args.resize:
            return cv2.resize(img, tuple(args.resize))
        s = preview_scale_factor()
        if abs(s - 1.0) > 1e-3:
            return cv2.resize(
                img, None, fx=s, fy=s,
                interpolation=cv2.INTER_AREA if s < 1.0 else cv2.INTER_LINEAR
            )
        return img

    cap = open_capture(args.video, args.backend)
    if not cap or not cap.isOpened():
        raise SystemExit(f"Could not open: {args.video}")

    # FPS used for selection timestamp and HUD display
    vid_fps = cap.get(cv2.CAP_PROP_FPS) or 0.0

    # ==== Pick a frame for selections ====
    selection_frame = None
    wanted_idx = None
    if args.select_at_frame is not None and args.select_at_frame >= 0:
        wanted_idx = int(args.select_at_frame)
    elif args.select_at_seconds is not None and vid_fps and vid_fps > 0:
        wanted_idx = max(0, int(round(float(args.select_at_seconds) * float(vid_fps))))

    if wanted_idx is not None:
        ok, f, _ = read_frame_at(cap, wanted_idx)
        if ok:
            selection_frame = f
        else:
            print(f"[WARN] Could not seek to frame {wanted_idx}; falling back to first visible frame.", file=sys.stderr)

    if selection_frame is None:
        ok_first, first = grab_first_visible_frame(
            cap, tries=args.first_frame_tries, brightness_thresh=args.first_frame_brightness, cam_warmup=args.cam_warmup
        )
        if not ok_first:
            print("[WARN] Could not find a bright first frame; proceeding.", file=sys.stderr)
        if first is None:
            raise SystemExit("Failed to read any frame from source.")
        selection_frame = first

    # Rotate selection frame for interactive picks
    selection_frame = rotate_frame(selection_frame, args.rotate)
    H_full, W_full = selection_frame.shape[:2]

    # MAIN ROI
    roi = None
    if args.roi:
        roi = clamp_roi(selection_frame, tuple(args.roi))
    elif args.select_roi:
        # --- IMPORTANT CHANGE ---
        # selection preview scale now includes global preview scale too
        sel_scale = float(args.roi_select_scale) * float(preview_scale_factor())
        pick = pick_roi_interactive(
            selection_frame,
            title="Select MAIN ROI (ENTER/SPACE=OK, C=Cancel)",
            scale=sel_scale
        )
        if pick: roi = clamp_roi(selection_frame, pick)

    if roi:
        x0, y0, w0, h0 = roi
        print(f"[MAIN ROI] top-left: x={x0}, y={y0}, w={w0}, h={h0}", flush=True)
    else:
        x0, y0, w0, h0 = 0, 0, W_full, H_full
        print(f"[MAIN ROI] None; using full frame: x=0, y=0, w={W_full}, h={H_full}", flush=True)

    # DIGITS perspective: pick quad
    digits_quad = None
    if args.digits_quad:
        pts = np.array(args.digits_quad, dtype=np.float32).reshape(4,2)
        digits_quad = order_quad(pts)
    elif args.select_digits_quad:
        q = pick_quad_interactive(selection_frame, title="Select DIGITS QUAD (ENTER=OK, R=reset, C=cancel)")
        if q is not None:
            digits_quad = q

    M_warp = None
    warp_size = None
    if digits_quad is not None:
        if args.digits_warp_size:
            warp_size = (int(args.digits_warp_size[0]), int(args.digits_warp_size[1]))
        M_warp, warp_size, digits_quad = compute_perspective(digits_quad, warp_size)
        print(f"[DIGITS QUAD] TL={digits_quad[0]} TR={digits_quad[1]} BR={digits_quad[2]} BL={digits_quad[3]}", flush=True)
        print(f"[DIGITS WARP] size={warp_size[0]}x{warp_size[1]}", flush=True)

    # DIGITS ROI selection (possibly in WARPED plane)
    digits_roi = None
    if M_warp is not None:
        warped_sel = cv2.warpPerspective(selection_frame, M_warp, warp_size)
        preview_for_roi = warped_sel
    else:
        preview_for_roi = selection_frame

    if args.digits_roi:
        digits_roi = clamp_roi(preview_for_roi, tuple(args.digits_roi))
    elif args.select_digits_roi:
        # optional: also scale this selection preview using global preview scale
        pick_led = pick_roi_interactive(
            preview_for_roi,
            title="Select DIGITS ROI (ENTER/SPACE=OK, C=Cancel)",
            scale=float(preview_scale_factor())
        )
        if pick_led:
            digits_roi = clamp_roi(preview_for_roi, pick_led)

    if M_warp is not None:
        if digits_roi:
            dx, dy, dw, dh = digits_roi
            print(f"[DIGITS ROI (WARPED)] x={dx}, y={dy}, w={dw}, h={dh}", flush=True)
        else:
            print("[DIGITS ROI (WARPED)] None; using full warped rect.", flush=True)
    else:
        if digits_roi:
            dx, dy, dw, dh = digits_roi
            print(f"[DIGITS ROI] top-left: x={dx}, y={dy}, w={dw}, h={dh}", flush=True)
        else:
            print("[DIGITS ROI] None selected; time_LED will be blank in CSV.", flush=True)

    # Rewind for processing
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    # FPS for overlay only
    use_fps = args.fps if args.fps > 0 else (vid_fps if vid_fps and vid_fps > 0 else 30.0)

    # Background for MAIN ROI
    if args.bg_method == "median":
        bg_f32 = build_bg_median(cap, (x0, y0, w0, h0), args.warmup, rotate_mode=args.rotate)
    else:
        ok, f0 = cap.read()
        if not ok: raise SystemExit("Could not read frame for background init.")
        f0 = rotate_frame(f0, args.rotate)
        f0 = f0[y0:y0+h0, x0:x0+w0]
        bg_f32 = init_bg_running(to_gray(f0))

    # Outputs
    writer = None
    if args.out is not None:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(args.out), fourcc, use_fps, (W_full, H_full))

    # === CSV path (auto if not provided) ===
    if args.csv is None:
        try:
            vpath = Path(args.video)
            csv_path = vpath.with_name(vpath.stem + "-track.csv")
        except Exception:
            csv_path = Path("track.csv")
        print(f"[CSV] --csv was not provided. Writing to: {csv_path}")
    else:
        csv_path = args.csv
        print(f"[CSV] Writing to: {csv_path}")

    csv_file = open(csv_path, "w", newline="")
    csv_writer = csv.writer(csv_file)

    # === CSV header section with ROI ===
    csv_writer.writerow(["ROI selection", "x", "y", "w", "h"])
    csv_writer.writerow(["", x0, y0, w0, h0])
    csv_writer.writerow(["time_LED", "x", "y"])

    # 7-seg debounce + hysteresis state
    last_val = None
    candidate_val = None
    candidate_count = 0
    prev_digit_patterns = None

    # ===== Cosine correction precompute =====
    cos_factor = 1.0
    if args.cam_distance and args.cam_distance > 0:
        theta = math.atan2(abs(args.cam_offset), float(args.cam_distance))
        c = math.cos(theta)
        if c <= 1e-9:
            print("[COS] Warning: angle too close to 90°, skipping correction.", file=sys.stderr)
            cos_factor = 1.0
        else:
            cos_factor = 1.0 / c
        print(f"[COS] cam_distance={args.cam_distance} m, cam_offset={args.cam_offset} m, "
              f"theta={math.degrees(theta):.3f} deg, factor={cos_factor:.6f}")

    # Prepare a resizable debug window for 7-seg if requested
    if args.show_digits:
        cv2.namedWindow("7seg mask (tune me)", cv2.WINDOW_NORMAL)

    # Main loop
    frame_idx = 0
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    proc_start = time.perf_counter()

    while True:
        ok, frame = cap.read()
        if not ok: break
        frame = rotate_frame(frame, args.rotate)

        # MAIN ROI view
        view = frame[y0:y0+h0, x0:x0+w0]
        roi_h = h0
        gray = to_gray(view)

        # Build masks depending on tracking mode
        if args.track_mode == "motion":
            mask_bin, diff_used, gray_proc = motion_mask(
                gray, bg_f32.astype(np.uint8), blur_k=args.blur, thr=args.thresh, r_open=args.open, r_close=args.close
            )
            if args.min_intensity > 0:
                gate = (gray_proc >= np.uint8(args.min_intensity)).astype(np.uint8) * 255
                mask_bin = cv2.bitwise_and(mask_bin, gate)
            if args.prefer_color != "none":
                color_mask = color_mask_hsv(view, args.prefer_color, args.color_s_min, args.color_v_min)
                if color_mask is not None:
                    mask_bin = cv2.bitwise_and(mask_bin, color_mask)
            weight = diff_used.astype(np.float32)
        else:
            if args.blur and args.blur % 2 == 1 and args.blur > 1:
                gray_proc = cv2.GaussianBlur(gray, (args.blur, args.blur), 0)
            else:
                gray_proc = gray
            mask_bin = brightest_mask(
                gray_proc,
                min_intensity=args.min_intensity,
                bright_thresh=args.bright_thresh,
                r_open=args.open,
                r_close=args.close
            )
            if args.prefer_color != "none":
                color_mask = color_mask_hsv(view, args.prefer_color, args.color_s_min, args.color_v_min)
                if color_mask is not None:
                    mask_bin = cv2.bitwise_and(mask_bin, color_mask)
            weight = gray_proc.astype(np.float32)
            diff_used = gray_proc

        # Pick component
        box_local, _ = pick_brightest_component(mask_bin, diff_used, bright_mask=None, weight=weight)

        cx_local = cy_local = math.nan
        if box_local is not None:
            bx, by, bw, bh = box_local
            cx_local = bx + bw / 2.0
            cy_local = by + bh / 2.0

        # 7-seg reading
        if M_warp is not None:
            img_for_digits = cv2.warpPerspective(frame, M_warp, warp_size)
        else:
            img_for_digits = frame

        if digits_roi is not None:
            dx, dy, dw, dh = clamp_roi(img_for_digits, tuple(digits_roi))
            droi = img_for_digits[dy:dy+dh, dx:dx+dw]
        else:
            droi = img_for_digits

        val, segmask, dboxes, patterns_now = read_digits_from_roi(
            droi,
            expected_n=args.digits,
            mode=args.digits_color,
            s_min=args.digits_s_min,
            v_min=args.digits_v_min,
            delta=args.digits_delta,
            polarity=args.digits_polarity,
            seg_on_thresh=args.seg_thresh,
            split_thresh_frac=args.digit_split_thresh,
            margin_frac=args.digit_margin,
            r_open=args.digits_open,
            r_close=args.digits_close,
            min_intensity=args.digits_min_intensity,
            allow_fewer=args.digits_allow_fewer,
            prev_patterns=prev_digit_patterns,
            seg_off_thresh=args.seg_off_thresh,
            seg_hyst=args.seg_hyst
        )
        prev_digit_patterns = patterns_now

        # 7-seg debug view
        if args.show_digits and segmask is not None:
            seg_debug = render_digits_debug(segmask, dboxes, val,
                                            seg_thresh=args.seg_thresh,
                                            show_seg_boxes=args.show_seg_boxes)
            dbg = apply_preview_resize(seg_debug)

            # fallback cap to ~420px width if huge and no resize/scale requested
            if (not args.resize) and (args.resize_scale is None) and (abs(float(args.display_scale) - 1.0) <= 1e-3):
                if dbg.shape[1] > 420:
                    scale = 420.0 / dbg.shape[1]
                    dbg = cv2.resize(dbg, (int(dbg.shape[1]*scale), int(dbg.shape[0]*scale)))

            cv2.imshow("7seg mask (tune me)", dbg)

        # Debounce + optional up-counter guard
        if val is not None:
            if candidate_val == val:
                candidate_count += 1
            else:
                candidate_val = val
                candidate_count = 1

            if candidate_count >= max(1, args.digits_debounce):
                proposed = candidate_val
                if args.enforce_upcounter and (last_val is not None):
                    if proposed == last_val or proposed == last_val + 1:
                        if proposed != last_val:
                            print(f"[7SEG] frame {frame_idx}: time_LED={proposed}")
                        last_val = proposed
                else:
                    if proposed != last_val:
                        print(f"[7SEG] frame {frame_idx}: time_LED={proposed}")
                    last_val = proposed

        # CSV write (origin bottom-left of ROI)
        time_led_str = str(int(last_val)) if last_val is not None else ""
        if math.isnan(cy_local):
            y_out = ""
        else:
            raw_y = (roi_h - 1 - cy_local)
            y_corr = raw_y * (cos_factor if (args.cam_distance and args.cam_distance > 0) else 1.0)
            y_out = f"{y_corr:.3f}"

        csv_writer.writerow([
            time_led_str,
            "" if math.isnan(cx_local) else f"{cx_local:.3f}",
            y_out
        ])

        # Overlays / output
        need_windows = args.display or args.show_filter or (writer is not None) or args.show_digits
        if need_windows:
            overlay = frame.copy()
            cv2.rectangle(overlay, (x0, y0), (x0+w0, y0+h0), (0, 255, 255), 2)
            if box_local is not None:
                bx, by, bw, bh = box_local
                cv2.rectangle(overlay, (bx + x0, by + y0), (bx + x0 + bw, by + y0 + bh), (255, 0, 0), 2)
                cv2.circle(overlay, (int(round(cx_local + x0)), int(round(cy_local + y0))), 3, (0,0,255), -1)

            if M_warp is not None and digits_quad is not None:
                pts = digits_quad.astype(np.int32)
                cv2.polylines(overlay, [pts], isClosed=True, color=(0, 200, 0), thickness=2)

            if (M_warp is None) and (digits_roi is not None):
                dx, dy, dw, dh = digits_roi
                cv2.rectangle(overlay, (dx, dy), (dx+dw, dy+dh), (0, 200, 0), 2)

            hud = f"frame={frame_idx} | mode={args.track_mode}"
            if last_val is not None:
                hud += f" | time_LED={last_val}"
            if args.cam_distance and args.cam_distance > 0:
                hud += f" | cos×={cos_factor:.3f}"
            cv2.putText(overlay, hud, (12, 28),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,0), 3, cv2.LINE_AA)
            cv2.putText(overlay, hud, (12, 28),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 1, cv2.LINE_AA)

            if writer is not None:
                writer.write(overlay)

            if args.display:
                disp = apply_preview_resize(overlay)
                cv2.imshow("Tracker + 7-seg Time", disp)

            if args.show_filter:
                vis_roi = cv2.bitwise_and(view, view, mask=mask_bin)
                filt_full = frame.copy()
                filt_full[y0:y0+h0, x0:x0+w0] = vis_roi
                filt_full = apply_preview_resize(filt_full)
                cv2.imshow(f"Filtered mask [{args.track_mode}]", filt_full)

            if (args.display or args.show_filter or args.show_digits) and (cv2.waitKey(1) & 0xFF in (27, ord('q'))):
                break

        frame_idx += 1

    # Timing summary
    elapsed = time.perf_counter() - proc_start
    eff_fps = (frame_idx / elapsed) if elapsed > 0 else float("inf")
    print(f"[DONE] Processed {frame_idx} frames in {elapsed:.3f} s ({eff_fps:.1f} FPS).", flush=True)

    # Cleanup
    cap.release()
    if writer:
        writer.release()
    csv_file.close()
    if args.display or args.show_filter or args.show_digits:
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
