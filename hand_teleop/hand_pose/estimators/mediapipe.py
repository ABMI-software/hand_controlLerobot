import os
from typing import List, Optional

import mediapipe as mp
import numpy as np
from mediapipe.tasks.python import vision

from hand_teleop.hand_pose.estimators.base import HandPoseEstimator
from hand_teleop.hand_pose.types import HandKeypointsPred, TrackedHandKeypoints


class MediaPipeEstimator(HandPoseEstimator):
    """
    Wraps MediaPipe GestureRecognizer in VIDEO mode and converts its world-
    landmarks to camera-space key-points (metres), matching HandKeypointsPred.
    """

    _DT_MS = 33  # advance timestamp by ~1 / 30 s per call

    def __init__(
        self,
        device: Optional[str] = None,
        model_path: str = os.path.join(os.path.dirname(__file__), "gesture_recognizer.task"),
        num_hands: int = 1,
    ):
        BaseOptions = mp.tasks.BaseOptions
        GestureRecognizerOptions = vision.GestureRecognizerOptions
        VisionRunningMode = vision.RunningMode

        options = GestureRecognizerOptions(
            base_options=BaseOptions(model_asset_path=model_path),
            running_mode=VisionRunningMode.VIDEO,
            num_hands=num_hands,
            min_hand_detection_confidence=0.1,
            min_hand_presence_confidence=0.4,
            min_tracking_confidence=0.4
        )
        self._rec = vision.GestureRecognizer.create_from_options(options)
        self._ts_ms = 0  # rolling video timestamp

    # ------------------------------------------------------------------ #
    def __call__(self, frame_rgb: np.ndarray, f_px: float) -> List[HandKeypointsPred]:
        h, w = frame_rgb.shape[:2]
        cx, cy = w * 0.5, h * 0.5

        mp_img = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=frame_rgb,
        )
        res = self._rec.recognize_for_video(mp_img, self._ts_ms)
        self._ts_ms += self._DT_MS

        if not res.hand_world_landmarks:
            return []

        preds: list[HandKeypointsPred] = []
        for wlm, ilm, hand in zip(
            res.hand_world_landmarks, res.hand_landmarks, res.handedness
        ):
            kp3d = np.array([[lm.x, -lm.y, lm.z] for lm in wlm], np.float32)
            kp2d = np.array([[lm.x * w, lm.y * h] for lm in ilm], np.float32)

            base3d = (kp3d[5] + kp3d[9]) * 0.5        # same origin trick
            base2d = (kp2d[5] + kp2d[9]) * 0.5

            d_world = np.linalg.norm(kp3d[5][:2] - kp3d[0][:2])
            d_pix   = np.linalg.norm(kp2d[5]       - kp2d[0])

            tz = f_px * d_world / d_pix
            tx = (base2d[0] - cx) * tz / f_px
            ty = (base2d[1] - cy) * tz / f_px
            t  = np.array([tx, -ty, tz], np.float32)

            kp_cam = kp3d - base3d + t

            preds.append(
                HandKeypointsPred(
                    is_right = hand[0].category_name.lower() == "right",
                    keypoints = TrackedHandKeypoints(
                        thumb_mcp   = kp_cam[2],
                        thumb_tip   = kp_cam[4],
                        index_base  = kp_cam[5],
                        index_tip   = kp_cam[8],
                        middle_base = kp_cam[9],
                        middle_tip  = kp_cam[12],
                    ),
                )
            )
        return preds