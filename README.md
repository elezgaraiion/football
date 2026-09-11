# Offside Detector — Week 1-2 Baseline

Pose estimation + persistent multi-object tracking on tactical-camera football clips.
This is the first working slice of a larger pipeline (pose → tracking → team
clustering → homography → offside line) that turns a single broadcast-style
camera into an automated offside check.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt