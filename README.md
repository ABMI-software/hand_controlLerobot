# Lerobot Controlled by hand

[![License](https://img.shields.io/github/license/ABMI-software/hand_controlLerobot)](LICENSE)
[![Python Version](https://img.shields.io/badge/python-3.6%2B-blue)](https://www.python.org/downloads/)

## Overview
A robot arm teleoperation system based on hand tracking via webcam. Lerobot Control enables intuitive control of the SO-101 robot through natural hand gestures, bridging the gap between human motion and robotic control with a seamless real-time interface.

### Key Benefits
- 🎯 Intuitive Control: Natural hand movements translate directly to robot actions
- ⚡ Real-time Response: Minimal latency between gesture recognition and robot movement
- 🔄 Flexible Tracking: Multiple tracking models available for different use cases
- 🛠 Customizable: Adjustable sensitivity and control parameters

## Table of Contents
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Configuration](#configuration)
- [Troubleshooting](#troubleshooting)
- [Contributing](#contributing)
- [License](#license)

## Features
- Real-time hand tracking and gesture recognition
- Control of robot joints and gripper
- Support for multiple tracking models (Wilor, MediaPipe)
- Adjustable parameters for sensitivity and control
- Safety limits and emergency stops
- Gesture recording and playback capabilities

## Installation

### Prerequisites
- Python 3.6 or higher
- Conda (recommended for managing dependencies)
- Webcam with minimum 720p resolution
- SO-101 robot hardware setup
- USB connection to the robot

### Installation Methods

#### Using Conda (Recommended)
1. Clone the repository:
   ```bash
   git clone https://github.com/ABMI-software/hand_controlLerobot.git
   cd hand_control
   ```

2. Create and activate environment using provided file:
   ```bash
   conda env create -f environment.yml
   conda activate hand_control
   ```

3. Install the package in development mode:
   ```bash
   pip install -e .
   ```

#### Using pip
1. Clone the repository:
   ```bash
   git clone https://github.com/ABMI-software/hand_controlLerobot.git
   cd hand_control
   ```

2. Create and activate a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use: venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   pip install -e .
   ```

#### Verify Installation
After installation, verify that everything is working:
```bash
python test_gripper_only.py
```

### Hardware Setup
1. Connect the SO-101 robot to your computer via USB
2. Ensure the webcam is properly connected and recognized
3. Position the webcam with a clear view of the control area

## Usage

### Basic Operation
1. Start only camera - hand tracking and pose estimation:
   ```bash
   python main.py
   ```
2. Start teleoperation with lerobot and Arducam:
   ```bash
   python3 poke_motor.py --hand right --model wilor --cam-idx 0 --fps 30   --so101-enable --so101-port /dev/serial/by-id/usb-1a86_USB_Single_Serial_5AA9018150-if00   --invert-z --raw --raw-min 1700 --raw-max 3200  --verbose
   ```

3. Start teleoperation with lerobot and astra depth camera:
   ```bash
   python hand_teleop_local.py --hand right --model wilor --cam-idx -1 --fps 30   --so101-enable --so101-port /dev/serial/by-id/usb-1a86_USB_Single_Serial_5AA9018150-if00   --invert-z --raw --raw-min 1700 --raw-max 3200 --verbose --print-joints 
   ```

4. Available control modes:
   - **Direct Control**: Control robot joints directly with hand movements
   - **Task Space**: Control end-effector position in Cartesian space
   - **Gripper Control**: Use pinch gesture to control gripper


5. to control wrist via mediapipe
    ```bash
    to be completed
    ```

### Advanced Features
- **Gesture Recording**: Save and replay common movement sequences
- **Safety Limits**: Built-in joint and velocity limits
- **Multiple Tracking Models**: Switch between different hand tracking models

## Configuration

### Tracking Settings
```bash
# Select tracking model (options: mediapipe, wilor)
python main.py --tracker mediapipe

# Adjust tracking sensitivity
python main.py --sensitivity 0.8
```

### Robot Settings
- Joint speed limits can be configured in `config/robot_config.yaml`
- Gesture mappings can be modified in `config/gesture_mapping.yaml`
- Camera calibration settings in `config/camera_config.yaml`

## Troubleshooting

### Common Issues
1. **Robot Not Detected**
   - Check USB connection
   - Verify correct port permissions
   - Run `python scan_bus.py` to detect connected devices

2. **Poor Tracking Performance**
   - Ensure good lighting conditions
   - Check webcam resolution settings
   - Try different tracking models

3. **Unexpected Robot Movement**
   - Verify calibration settings
   - Check gesture sensitivity settings
   - Ensure clean background for better tracking

## Contributing
1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.