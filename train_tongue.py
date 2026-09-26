"""
Train the tongue classifier from tongue_data/ (made by record_tongue.py).

    python train_tongue.py

Prints cross-validated accuracy + a confusion matrix, saves tongue_model.pkl.
"""
import os

import cv2
import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from face import CLASSIFIER, DATA_DIR, LABELS, embed, make_embedder

embedder = make_embedder()
X, y = [], []
for idx, label in enumerate(LABELS):
    files = os.listdir(os.path.join(DATA_DIR, label))
    print(f"{label}: {len(files)} images")
    for f in files:
        X.append(embed(embedder, cv2.imread(os.path.join(DATA_DIR, label, f))))
        y.append(idx)
X, y = np.array(X), np.array(y)

# Embeddings are L2-normalised (tiny values); scale them or the model stays
# timid and never gets confident enough to cross p.py's thresholds.
clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, C=0.1))
# Unshuffled folds: neighbouring frames are near-duplicates, so this is a more
# honest estimate than a random split.
pred = cross_val_predict(clf, X, y, cv=5)
print(f"\ncross-val accuracy: {(pred == y).mean():.1%}")
print("confusion (rows = true, cols = predicted):", LABELS)
print(confusion_matrix(y, pred))

clf.fit(X, y)
joblib.dump(clf, CLASSIFIER)
print(f"\nsaved {CLASSIFIER}")
