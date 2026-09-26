"""
Face-gesture TikTok navigator (MVP), driven through Playwright.

Setup:
    pip install mediapipe opencv-python playwright
Run:
    python p.py
The script opens its own Chrome window (separate profile, so log into TikTok
once there) and controls the TikTok tab directly -- it doesn't need focus.

Gestures (tune thresholds using the on-screen scores):
    Open mouth      -> next video   (scroll down)
    Raise eyebrows  -> prev video   (scroll up)
    Big smile       -> pause / play
Press q in the preview window to quit.
"""
import os
import socket
import subprocess
import time
import urllib.request

import cv2
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision
from playwright.sync_api import sync_playwright

# --- browser -----------------------------------------------------------------
CHROME = r"C:\Program Files\Google\Chrome\Application\chrome.exe"
PROFILE = os.path.join(os.environ["LOCALAPPDATA"], "tiktok-chrome")
PORT = 9222
TIKTOK = "https://www.tiktok.com/foryou"


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

pw = sync_playwright().start()
browser = pw.chromium.connect_over_cdp(f"http://127.0.0.1:{PORT}")
ctx = browser.contexts[0]
page = next((p for p in ctx.pages if "tiktok.com" in p.url), None)
if page is None:
    page = ctx.new_page()
    page.goto(TIKTOK)

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


ACTIONS = {
    "down":  lambda: nav("next", "ArrowDown"),
    "up":    lambda: nav("prev", "ArrowUp"),
    "pause": lambda: print("  ->", page.evaluate(TOGGLE_JS)),
}

# --- face model --------------------------------------------------------------
MODEL = "face_landmarker.task"
URL = ("https://storage.googleapis.com/mediapipe-models/face_landmarker/"
       "face_landmarker/float16/1/face_landmarker.task")
if not os.path.exists(MODEL):
    print("Downloading model (~4 MB)...")
    urllib.request.urlretrieve(URL, MODEL)

landmarker = vision.FaceLandmarker.create_from_options(
    vision.FaceLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=MODEL),
        running_mode=vision.RunningMode.VIDEO,
        output_face_blendshapes=True,
        num_faces=1,
    )
)

# (name, score function over blendshapes, threshold, action)
GESTURES = [
    ("next (mouth open)", lambda s: s["jawOpen"], 0.50, "down"),
    ("prev (brows up)",   lambda s: s["browInnerUp"], 0.60, "up"),
    ("pause (smile)",     lambda s: (s["mouthSmileLeft"] + s["mouthSmileRight"]) / 2, 0.70, "pause"),
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
