# tongue_os

Control TikTok, Blackjack and Swipe Club hands-free with your tongue and winks.

You need Windows, Google Chrome, a webcam and Python.

## Run

```
pip install mediapipe opencv-python playwright scikit-learn
python start.py
```

A Chrome window opens with TikTok, [Blackjack](https://blackjack-coral.vercel.app/)
and [Swipe Club](https://swipeapptest2.vercel.app/), starting on TikTok.
A webcam window shows your live gesture scores. Press `q` in it to quit.

## Controls

| Gesture | TikTok | Blackjack | Swipe Club |
|---|---|---|---|
| Tongue up | Previous video | Hit | Yes |
| Tongue down | Next video | Stand | No |
| Wink right / left | Next / previous tab | same | same |

Relax your face between gestures. Each gesture fires once until you do.

## Train your own model (optional)

A trained tongue model is included. If it doesn't pick up your tongue well,
train your own (about 10 minutes):

```
python record_tongue.py   # record your tongue poses (instructions on screen)
python train_tongue.py    # train the model
```

## Tips

- **Too sensitive or not sensitive enough:** change the thresholds in
  `GESTURES` in `start.py`. A score turns green in the webcam window when it
  passes its threshold.
- **Left and right winks feel swapped:** swap the two wink names in `GESTURES`.
- **Freezes on startup:** fully close Chrome, including from the system tray,
  and run `start.py` again.
