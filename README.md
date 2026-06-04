# Hand Cricket - MediaPipe Hand Tracking

A simple hand-cricket game driven by real-time hand landmark detection using MediaPipe and OpenCV.

## Requirements
- Python 3.9+ recommended
- A webcam

## Setup
1. Create and activate a virtual environment (optional but recommended).
2. Install dependencies:

```bash
pip install -r requirements.txt
```

## Run
Ensure the model file `hand_landmarker.task` is in the same folder, then run:

```bash
python HandTracking.py
```

## Controls
- Press `q` to quit.

## Notes
- The camera feed opens in full-screen mode.
- If the model path changes, update `model_asset_path` in `HandTracking.py`.
