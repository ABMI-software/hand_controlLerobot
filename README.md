# hand-teleop

**hand-teleop** turns webcam input into robot joint positions in real time.
Built for [LeRobot](https://github.com/huggingface/lerobot), it runs directly in your Python process — no server required.

## Features

* **Wilor** (GPU) and **MediaPose** (CPU) support
* Direct joint output (no need for separate IK)
* Kalman filter smoothing for jitter-free movement
* Minimal latency, runs separately in background thread
* easy to get started, with one line pip installation
* built-in monkey patch for lerobot integration

## Installation

### GPU-based (Wilor – accurate, requires CUDA)

```bash
pip install "hand-teleop @ git+https://github.com/joeclinton1/hand-teleop.git#egg=hand-teleop[wilor]"
```

### CPU-based (MediaPipe – lightweight)

```bash
pip install "hand-teleop @ git+https://github.com/joeclinton1/hand-teleop.git#egg=hand-teleop[mediapipe]"
```

### Development install

```bash
# Create and activate a clean Python 3.10 environment
conda create -n hand-teleop python=3.10 -y
conda activate hand-teleop

# (Optional) Install ffmpeg if needed for webcam/video support
conda install -c conda-forge ffmpeg

# Clone the repo and install in editable mode
git clone https://github.com/joeclinton1/hand-teleop.git
cd hand-teleop
pip install -e ".[wilor]"  # or ".[mediapipe]"
```

### Optional: Enable `read_hand_state_joint` (requires Pinocchio)

If you want to use `read_hand_state_joint` or any functionality that relies on forward/inverse kinematics, you'll need to install the `pinocchio` library separately, as it's not available on PyPI and cannot be managed directly by Poetry.

```bash
conda install -c conda-forge pinocchio
````

This step is **only required** if you're using joint-space kinematics features. All other functionality will work without it. If `pinocchio` is missing and you attempt to use these features, you'll get an error message.


## Basic Usage

```python
from hand_teleop import HandTracker

tracker = HandTracker(model="mediapipe", hand="right", cam_idx=0)
tracker.start()

while True:
    joints = tracker.get_joint_positions()
    robot.set_joint_positions(joints)
```

---

## Running the Demo

To quickly test hand tracking with your webcam, run:

```bash
python main.py  # add --quiet to suppress output
```

This opens your webcam and prints hand pose and FPS. Press `q` or `Esc` to exit.

---

## Keyboard Controls

* `p` – Pause/resume
* `space` – Realign temporarily (hold to pause tracking)

---

## License

Apache 2.0