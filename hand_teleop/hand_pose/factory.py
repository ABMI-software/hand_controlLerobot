from typing import Callable, Literal

ModelName = Literal["wilor", "mediapipe", "apriltag"]

def _lazy_import(module_path: str, class_name: str, install_hint: str) -> Callable:
    def _constructor(**kwargs):
        try:
            module = __import__(module_path, fromlist=[class_name])
            cls = getattr(module, class_name)
            return cls(**kwargs)
        except ImportError as e:
            raise ImportError(
                f"{class_name} was requested but required package is not installed.\n"
                f"Run: {install_hint}"
            ) from e
    return _constructor

_REGISTRY: dict[str, Callable] = {
    "wilor": _lazy_import(
        "hand_teleop.hand_pose.estimators.wilor",
        "WiLorEstimator",
        "pip install 'https://github.com/Joeclinton1/WiLoR-mini'"
    ),
    "mediapipe": _lazy_import(
        "hand_teleop.hand_pose.estimators.mediapipe",
        "MediaPipeEstimator",
        "pip install mediapipe"
    ),
    "apriltag": _lazy_import(
        "hand_teleop.hand_pose.estimators.apriltag",
        "AprilTagCubeEstimator",
        "pip install pupil-apriltags"
    ),
}

def create_estimator(name: ModelName = "wilor", device=None, **kwargs):
    if name not in _REGISTRY:
        raise ValueError(f"Unknown estimator '{name}'. Available: {list(_REGISTRY)}")
    return _REGISTRY[name](device=device, **kwargs)