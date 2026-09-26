"""Shared face / mouth helpers for start.py, record_tongue.py and train_tongue.py."""
import os
import urllib.request

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

HERE = os.path.dirname(os.path.abspath(__file__))
LANDMARK_MODEL = os.path.join(HERE, "face_landmarker.task")
LANDMARK_URL = ("https://storage.googleapis.com/mediapipe-models/face_landmarker/"
                "face_landmarker/float16/1/face_landmarker.task")
EMBED_MODEL = os.path.join(HERE, "mobilenet_v3_small.tflite")
EMBED_URL = ("https://storage.googleapis.com/mediapipe-models/image_embedder/"
             "mobilenet_v3_small/float32/1/mobilenet_v3_small.tflite")

LABELS = ["neutral", "tongue_up", "tongue_down"]
DATA_DIR = os.path.join(HERE, "tongue_data")
CLASSIFIER = os.path.join(HERE, "tongue_model.pkl")


def _download(path, url):
    if not os.path.exists(path):
        print(f"Downloading {os.path.basename(path)}...")
        urllib.request.urlretrieve(url, path)


def make_landmarker():
    _download(LANDMARK_MODEL, LANDMARK_URL)
    return vision.FaceLandmarker.create_from_options(
        vision.FaceLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=LANDMARK_MODEL),
            running_mode=vision.RunningMode.VIDEO,
            output_face_blendshapes=True,
            num_faces=1,
        )
    )


def make_embedder():
    _download(EMBED_MODEL, EMBED_URL)
    return vision.ImageEmbedder.create_from_options(
        vision.ImageEmbedderOptions(
            base_options=mp_python.BaseOptions(model_asset_path=EMBED_MODEL),
            l2_normalize=True,
        )
    )


def crop_mouth(frame, landmarks, size=128):
    """Square crop centred on the mouth, scaled by eye distance so it covers
    nose-to-chin regardless of how close you sit. Returns BGR image or None."""
    h, w = frame.shape[:2]
    pt = lambda i: np.array([landmarks[i].x * w, landmarks[i].y * h])
    center = (pt(61) + pt(291) + pt(0) + pt(17)) / 4    # lip corners, top, bottom
    half = 0.55 * np.linalg.norm(pt(33) - pt(263))       # outer eye corners
    x0, y0 = (center - half).astype(int)
    x1, y1 = (center + half).astype(int)
    if x0 < 0 or y0 < 0 or x1 > w or y1 > h:
        return None
    return cv2.resize(frame[y0:y1, x0:x1], (size, size))


def embed(embedder, crop_bgr):
    rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
    result = embedder.embed(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb))
    return np.array(result.embeddings[0].embedding)
