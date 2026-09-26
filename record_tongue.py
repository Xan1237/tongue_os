"""
Record mouth crops for the tongue classifier.

    python record_tongue.py

Press a key to START saving every frame as that label, press it again
(or space) to STOP:
    1  neutral      - normal face, talking, smiling, mouth open, looking around
    2  tongue_up    - tongue out, pointing up toward your nose
    3  tongue_down  - tongue out, pointing down toward your chin
Get into the pose before starting. Aim for 300+ per label (~10-15 s each). Move your head a little and vary lighting.
q quits. Run again any time to add more; images accumulate.
"""
import os
import time

import cv2
import mediapipe as mp

from face import DATA_DIR, LABELS, crop_mouth, make_landmarker

KEYS = {ord(str(i + 1)): label for i, label in enumerate(LABELS)}
for label in LABELS:
    os.makedirs(os.path.join(DATA_DIR, label), exist_ok=True)
count = lambda label: len(os.listdir(os.path.join(DATA_DIR, label)))

landmarker = make_landmarker()
cap = cv2.VideoCapture(0)
t0 = time.time()
label = None     # label currently being recorded
while cap.isOpened():
    ok, frame = cap.read()
    if not ok:
        break
    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = landmarker.detect_for_video(
        mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb),
        int((time.time() - t0) * 1000))

    crop = crop_mouth(frame, result.face_landmarks[0]) if result.face_landmarks else None
    if crop is not None:
        frame[10:138, -138:-10] = crop           # preview of what gets saved
        if label:
            name = f"{int(time.time() * 1000)}.png"
            cv2.imwrite(os.path.join(DATA_DIR, label, name), crop)

    for i, l in enumerate(LABELS):
        color = (0, 255, 0) if l == label else (255, 255, 255)
        cv2.putText(frame, f"{i + 1} {l}: {count(l)}", (10, 30 + 30 * i),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
    if crop is None:
        cv2.putText(frame, "no face / mouth off-screen", (10, 130),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    cv2.imshow("record tongue (1/2/3 start/stop, space stop, q quit)", frame)
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break
    if key == ord(" "):
        label = None
    elif key in KEYS:
        label = None if label == KEYS[key] else KEYS[key]

cap.release()
cv2.destroyAllWindows()
