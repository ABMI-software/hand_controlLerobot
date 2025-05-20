from typing import Callable, Literal

ModelName = Literal["wilor", "mediapipe"]

_REGISTRY: dict[str, Callable] = {}

def _lazy_import_wilor():
    try:
        from hand_teleop.hand_pose.estimators.wilor import WiLorEstimator
        return WiLorEstimator
    except ImportError as e:
        raise ImportError(
            "WilorEstimator was requested but `wilor-mini` is not installed.\n"
            "Run: `pip install 'https://github.com/Joeclinton1/WiLoR-mini'` or use the '[wilor]' extra."
        ) from e


def _lazy_import_mediapipe():
    try:
        from hand_teleop.hand_pose.estimators.mediapipe import MediaPipeEstimator
        return MediaPipeEstimator
    except ImportError as e:
        raise ImportError(
            "MediaPipeEstimator was requested but `mediapipe` is not installed.\n"
            "Run: `pip install mediapipe` or use the '[mediapipe]' extra."
        ) from e

_REGISTRY["wilor"] = _lazy_import_wilor
_REGISTRY["mediapipe"] = _lazy_import_mediapipe

def create_estimator(
    name: ModelName = "wilor",
    device=None,
    **kwargs
):
    if name not in _REGISTRY:
        raise ValueError(f"Unknown estimator '{name}'. Available: {list(_REGISTRY)}")

    estimator_class = _REGISTRY[name]()
    return estimator_class(device=device, **kwargs)