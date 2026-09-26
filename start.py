"""
Face-gesture TikTok navigator (MVP), driven through Playwright.

Setup:
    pip install mediapipe opencv-python playwright scikit-learn
    python record_tongue.py    # record your tongue poses
    python train_tongue.py     # -> tongue_model.pkl
Run:
    python start.py
The script opens its own Chrome window (separate profile, so log into TikTok
once there) and controls the TikTok tab directly -- it doesn't need focus.

Gestures (tune thresholds using the on-screen scores):
    Tongue down  -> next video   (scroll down)  | blackjack: S | swipe club: N
    Tongue up    -> prev video   (scroll up)    | blackjack: H | swipe club: Y
    Wink right   -> next browser tab
    Wink left    -> previous browser tab
Press q in the preview window to quit.
"""
import collections
import json
import os
import socket
import subprocess
import sys
import time
import urllib.request

import cv2
import joblib
import mediapipe as mp
import numpy as np
from playwright.sync_api import sync_playwright

from face import CLASSIFIER, LABELS, crop_mouth, embed, make_embedder, make_landmarker

if not os.path.exists(CLASSIFIER):
    sys.exit("No tongue_model.pkl yet: run record_tongue.py, then train_tongue.py.")

# --- browser -----------------------------------------------------------------
CHROME = r"C:\Program Files\Google\Chrome\Application\chrome.exe"
PROFILE = os.path.join(os.environ["LOCALAPPDATA"], "tiktok-chrome")
PORT = 9222
TIKTOK = "https://www.tiktok.com/foryou"
# Extra tabs opened on launch: url -> (tongue-up key, tongue-down key).
SITES = {
    "https://blackjack-coral.vercel.app/": ("h", "s"),   # hit / stand
    "https://swipeapptest2.vercel.app/": ("y", "n"),     # swipe club
}


def port_open():
    with socket.socket() as s:
        return s.connect_ex(("127.0.0.1", PORT)) == 0


if not port_open():
    subprocess.Popen([CHROME, f"--remote-debugging-port={PORT}",
                      f"--user-data-dir={PROFILE}", TIKTOK])
    for _ in range(50):
        if port_open():
            break
        time.sleep(0.2)

# Chrome can keep running with every window closed; Playwright then hangs on
# connect, so give it a tab first.
tabs = json.load(urllib.request.urlopen(f"http://127.0.0.1:{PORT}/json/list"))
if not any(t["type"] == "page" for t in tabs):
    urllib.request.urlopen(urllib.request.Request(
        f"http://127.0.0.1:{PORT}/json/new?{TIKTOK}", method="PUT"))

pw = sync_playwright().start()
browser = pw.chromium.connect_over_cdp(f"http://127.0.0.1:{PORT}")
ctx = browser.contexts[0]
page = next((p for p in ctx.pages if "tiktok.com" in p.url), None)
if page is None:
    page = ctx.new_page()
    page.goto(TIKTOK)

# Open the extra site tabs alongside TikTok, but start on TikTok.
for url in SITES:
    if not any(p.url.startswith(url) for p in ctx.pages):
        try:
            ctx.new_page().goto(url)
        except Exception as e:
            print(f"not reachable: {url}:", e)
page.bring_to_front()

# Toggle whichever <video> takes up the most of the viewport.
TOGGLE_JS = """() => {
  let best = null, bestArea = 0;
  for (const v of document.querySelectorAll('video')) {
    const r = v.getBoundingClientRect();
    const w = Math.max(0, Math.min(r.right, innerWidth) - Math.max(r.left, 0));
    const h = Math.max(0, Math.min(r.bottom, innerHeight) - Math.max(r.top, 0));
    if (w * h > bestArea) { bestArea = w * h; best = v; }
  }
  if (!best) return 'no video';
  if (best.paused) { best.play(); return 'play'; }
  best.pause(); return 'pause';
}"""

# Click TikTok's own feed arrows from JS so popups/overlays can't swallow it.
NAV_JS = """(dir) => {
  const b = document.querySelector(`[data-e2e="feed-navigation-${dir}"]`);
  if (!b || b.disabled) return false;
  b.click(); return true;
}"""


def nav(direction, key):
    if not page.evaluate(NAV_JS, direction):
        page.keyboard.press(key)


def app_tabs():
    """TikTok, then each SITES tab, in that fixed order (the order they open in).
    Other tabs are skipped. Chrome doesn't expose the tab strip's visual order."""
    tabs = [next((p for p in ctx.pages if "tiktok.com" in p.url), None)]
    tabs += [next((p for p in ctx.pages if p.url.startswith(u)), None) for u in SITES]
    return [t for t in tabs if t]


def switch_tab(step):
    """Move right (+1) / left (-1) through app_tabs(); later gestures act on it."""
    global page
    tabs = app_tabs()
    i = tabs.index(page) if page in tabs else 0
    page = tabs[(i + step) % len(tabs)]
    page.bring_to_front()
    print("  ->", page.url[:80])


def tongue(which, direction, key):
    """On a SITES tab press its key (which: 0 = up, 1 = down); else navigate the feed."""
    site = next((keys for url, keys in SITES.items() if page.url.startswith(url)), None)
    if site:
        page.keyboard.press(site[which])
    else:
        nav(direction, key)


ACTIONS = {
    "down":  lambda: tongue(1, "next", "ArrowDown"),
    "up":    lambda: tongue(0, "prev", "ArrowUp"),
    "tab_right": lambda: switch_tab(+1),
    "tab_left":  lambda: switch_tab(-1),
    "pause": lambda: print("  ->", page.evaluate(TOGGLE_JS)),
}

# --- face + tongue models ----------------------------------------------------
landmarker = make_landmarker()
embedder = make_embedder()
tongue_clf = joblib.load(CLASSIFIER)
# Average tongue probabilities over the last few frames so one misread frame
# can't scroll the feed.
SMOOTH = 5
tongue_hist = collections.deque(maxlen=SMOOTH)

# (name, score function over blendshapes + tongue probs, threshold, action)
GESTURES = [
    ("down: next/S/N (tongue)", lambda s: s["tongue_down"], 0.40, "down"),
    ("up: prev/H/Y (tongue)", lambda s: s["tongue_up"], 0.60, "up"),
    # Wink = one eye shut while the other stays open (normal blinks cancel out).
    # The preview is mirrored, so MediaPipe's eyeBlinkLeft is your right eye.
    ("tab right (wink R)", lambda s: s["eyeBlinkLeft"] - s["eyeBlinkRight"], 0.20, "tab_right"),
    ("tab left (wink L)",  lambda s: s["eyeBlinkRight"] - s["eyeBlinkLeft"], 0.20, "tab_left"),
]
COOLDOWN = 1.0   # seconds between actions
armed = True     # must return to neutral before the next gesture fires
last_fire = 0.0

cap = cv2.VideoCapture(0)
t0 = time.time()
while cap.isOpened():
    ok, frame = cap.read()
    if not ok:
        break
    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    result = landmarker.detect_for_video(image, int((time.time() - t0) * 1000))

    if result.face_blendshapes:
        s = {c.category_name: c.score for c in result.face_blendshapes[0]}
        crop = crop_mouth(frame, result.face_landmarks[0])
        if crop is not None:
            tongue_hist.append(tongue_clf.predict_proba([embed(embedder, crop)])[0])
        probs = np.mean(tongue_hist, axis=0) if tongue_hist else np.zeros(len(LABELS))
        s.update(zip(LABELS, probs))
        vals = [(name, fn(s), th, act) for name, fn, th, act in GESTURES]
        active = [v for v in vals if v[1] > v[2]]
        now = time.time()

        if not active:
            armed = True
        elif armed and now - last_fire > COOLDOWN:
            name, _, _, act = max(active, key=lambda v: v[1] - v[2])
            print("fired:", name)
            try:
                ACTIONS[act]()
            except Exception as e:  # tab closed / navigating, keep going
                print("  action failed:", e)
            last_fire, armed = now, False

        for i, (name, v, th, _) in enumerate(vals):
            color = (0, 255, 0) if v > th else (0, 0, 255)
            cv2.putText(frame, f"{name}: {v:.2f} (>{th})", (10, 30 + 30 * i),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

    cv2.imshow("face control (q to quit)", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
pw.stop()
