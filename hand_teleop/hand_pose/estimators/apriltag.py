from __future__ import annotations

from typing import Dict, List, Tuple

import cv2
import numpy as np
from pupil_apriltags import Detector

from hand_teleop.hand_pose.estimators.base import HandPoseEstimator
from hand_teleop.hand_pose.types import HandKeypointsPred, TrackedHandKeypoints

# ─── Physical constants ───────────────────────────────────────────────────
_TAG_SIZE  = 0.018
_CUBE_SIZE = 0.025

# ─── Image prep ───────────────────────────────────────────────────────────
def preprocess(gray: np.ndarray) -> np.ndarray:
    clahe = cv2.createCLAHE(3.0, (8, 8))
    gray  = clahe.apply(gray)
    gray  = cv2.convertScaleAbs(gray, alpha=1.2, beta=15)
    return cv2.bilateralFilter(gray, 3, 10, 10)

# ─── Drawing helpers ──────────────────────────────────────────────────────
def _proj(pt3, fx, fy, cx, cy):
    return int(fx * pt3[0] / pt3[2] + cx), int(fy * pt3[1] / pt3[2] + cy)


def draw_coloured_cube_pose(frame,
                            centre, fwd, left, up,
                            fx, fy, cx, cy,
                            size=_CUBE_SIZE, thick=1, alpha=0.5):
    """
    Rubik-style cube with back-face culling and 0.5-alpha blending.
    Face colours: front-white, back-yellow, left-orange, right-red,
                  top-green, bottom-blue
    """
    h  = size * 0.5
    fc = centre - fwd * h         # front-face centre
    bc = centre + fwd * h         # back-face centre

    # eight 3-D corners
    C = [
        fc + left * h + up * h,  # 0
        fc - left * h + up * h,  # 1
        fc - left * h - up * h,  # 2
        fc + left * h - up * h,  # 3
        bc + left * h + up * h,  # 4
        bc - left * h + up * h,  # 5
        bc - left * h - up * h,  # 6
        bc + left * h - up * h,  # 7
    ]
    pts = [_proj(p, fx, fy, cx, cy) for p in C]

    # faces (CCW order when viewed from outside)
    faces = [
        (0, 1, 2, 3), (7, 6, 5, 4), (3, 7, 4, 0),
        (1, 5, 6, 2), (0, 4, 5, 1), (2, 6, 7, 3),
    ]
    colours = [
        (255,255,255), (0,255,255), (0,128,255),
        (0,0,255), (0,255,0), (255,0,0),
    ]

    overlay = frame.copy()
    for f, col in zip(faces, colours):
        a, b, c = C[f[0]], C[f[1]], C[f[2]]
        normal  = np.cross(b - a, c - a)
        face_center = sum(C[i] for i in f) / 4
        view_vec = face_center / np.linalg.norm(face_center)
        if np.dot(normal, view_vec) >= 0:     # back-face, skip
            continue

        poly = np.array([pts[i] for i in f], np.int32)
        cv2.fillConvexPoly(overlay, poly, col)
        for i in range(4):
            cv2.line(overlay, pts[f[i]], pts[f[(i+1)%4]], (0,0,0), thick)

    cv2.addWeighted(overlay, alpha, frame, 1-alpha, 0, frame)


def draw_axes(frame, centre, fwd, left, up, fx, fy, cx, cy,
              scale=0.05, thick=2):
    for v, col in [(fwd,(0,0,255)), (up,(0,255,0)), (left,(255,0,0))]:
        start = np.array([fx*centre[0]/centre[2]+cx,
                          fy*centre[1]/centre[2]+cy]).astype(int)
        end3d = centre + v*scale
        end   = np.array([fx*end3d[0]/end3d[2]+cx,
                          fy*end3d[1]/end3d[2]+cy]).astype(int)
        cv2.line(frame, tuple(start), tuple(end), col, thick)

def draw_hand_keypoints(frame, hand_preds, fx, fy, cx, cy,
                        radius=4, colour=(255, 0, 255)):
    """Draw every 3-D key-point as a little magenta dot.

    The model internally uses a flipped coordinate system (negated x and y),
    so we unflip the points here for correct visualization in image space.
    """
    for pred in hand_preds:
        for pt3 in pred.keypoints.__dict__.values():
            pt3_unflipped = AprilTagCubeEstimator._unflip(pt3)
            u, v = _proj(pt3_unflipped, fx, fy, cx, cy)
            cv2.circle(frame, (u, v), radius, colour, -1)

def annotate_frame(frame, poses, tag_dets, hand_preds, cam_p):
    """
    Single entry-point for *all* visual debug: cubes, axes, tag outlines,
    and the new key-points.
    """
    fx, fy, cx, cy = cam_p

    for cube, det in tag_dets.items():
        centre, fwd, left, up = poses[cube]
        draw_coloured_cube_pose(frame, centre, fwd, left, up, fx, fy, cx, cy)
        draw_axes(frame, centre, fwd, left, up, fx, fy, cx, cy)

        # tag outline (optional)
        pts = det.corners.astype(int)
        for i in range(4):
            cv2.line(frame, pts[i], pts[(i+1)%4], (0,255,0), 1)
        cv2.putText(frame, str(det.tag_id), tuple(det.center.astype(int)),
                    cv2.FONT_HERSHEY_SIMPLEX, .5, (255,0,0), 1)

    draw_hand_keypoints(frame, hand_preds, fx, fy, cx, cy)

# ─── Estimator class (unchanged) ──────────────────────────────────────────
class AprilTagCubeEstimator(HandPoseEstimator):
    _TH_OFFS = -0.02

    def __init__(self, **kw):
        super().__init__()
        self.thumb_length  = kw.get("thumb_length", 0.08)
        self.knuckle_width = kw.get("knuckle_width", 0.025)
        self._MID_TIP_FWD  = self.knuckle_width - 0.005

        self._det = Detector(
            families      = kw.get("families", "tag25h9"),
            nthreads      = kw.get("n_threads", 4),
            quad_decimate = kw.get("quad_decimate", 1.0),
            quad_sigma    = kw.get("quad_sigma", 0.0)
        )

        self._prev_pose = {"index": None, "thumb": None}
        self._default_pose = (
            np.array([0, 0, 0.15], np.float32),
            np.array([0, 0, -1],   np.float32),
            np.array([-1, 0, 0],   np.float32),
            np.array([0, -1, 0],   np.float32)
        )

    @staticmethod
    def _make_pose(det, extra_fwd: float = 0.0) -> Tuple[np.ndarray, ...]:
        R, t = det.pose_R, det.pose_t.flatten()
        x, y, z = R[:, 0], R[:, 1], R[:, 2]  # tag axes in camera frame

        face = det.tag_id % 5
        if   face == 0:                       # FRONT
            fwd, left, up = -z, -x, -y
        elif face == 1:                       # LEFT
            fwd, left, up =  x, -z, -y
        elif face == 2:                       # RIGHT
            fwd, left, up = -x,  z, -y
        elif face == 3:                       # TOP
            fwd, left, up =  y, -x, -z
        else:                                 # BOTTOM
            fwd, left, up = -y, -x,  z

        fwd, left, up = [v / np.linalg.norm(v) for v in (fwd, left, up)]

        # **same maths as your original**
        offset = -z / np.linalg.norm(z)       # i.e. face-0 fwd
        centre = t + offset * (-_CUBE_SIZE * 0.5 + extra_fwd)

        return centre, fwd, left, up

    @staticmethod
    def _unflip(p): return p.copy() * [-1, -1, 1]

    def detect_cubes(self, frame_rgb: np.ndarray, focal: float,
                     use_thumb_offset: bool = True
                     ) -> Tuple[Dict[str,Tuple[np.ndarray,...]],
                                Dict[str,any], np.ndarray]:
        gray  = preprocess(cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2GRAY))
        h, w  = gray.shape
        cam_p = (focal, focal, w*.5, h*.5)
        dets  = self._det.detect(gray, True, camera_params=cam_p,
                                 tag_size=_TAG_SIZE)

        best = {}
        for d in dets:
            cube = "index" if d.tag_id < 5 else "thumb"
            if cube not in best or d.decision_margin > best[cube].decision_margin:
                best[cube] = d

        poses = {}
        for cube in ("index", "thumb"):
            if cube in best:
                off = self._TH_OFFS if cube=="thumb" and use_thumb_offset else 0.0
                poses[cube] = self._make_pose(best[cube], off)
                self._prev_pose[cube] = poses[cube]
            else:
                poses[cube] = self._prev_pose[cube] or self._default_pose

        return poses, best, frame_rgb

    def __call__(self, frame_rgb: np.ndarray, focal: float
                 ) -> List[HandKeypointsPred]:
        frame_rgb = cv2.flip(frame_rgb, 1)
        poses, _, _ = self.detect_cubes(frame_rgb, focal)

        ic,  ifwd, *_        = poses["index"]
        tc,  tfwd, tlft, tup = poses["thumb"]

        thumb_tip   = tc
        index_tip   = ic
        thumb_mcp   = thumb_tip  + tfwd * -self.thumb_length
        index_base  = thumb_mcp  + tup   *  self.knuckle_width
        middle_base = thumb_mcp  + tlft  *  self.knuckle_width
        middle_tip  = index_tip  + tfwd  *  self._MID_TIP_FWD

        return [HandKeypointsPred(
            is_right=True,
            keypoints=TrackedHandKeypoints(
                thumb_mcp   = self._unflip(thumb_mcp).astype(np.float32),
                thumb_tip   = self._unflip(thumb_tip).astype(np.float32),
                index_base  = self._unflip(index_base).astype(np.float32),
                index_tip   = self._unflip(index_tip).astype(np.float32),
                middle_base = self._unflip(middle_base).astype(np.float32),
                middle_tip  = self._unflip(middle_tip).astype(np.float32),
            )
        )]

# ─── Live viewer ──────────────────────────────────────────────────────────
def main(cam_idx=0, quad_decimate=1.0, quad_sigma=0.0):
    print("starting video capture")
    cap = cv2.VideoCapture(cam_idx, cv2.CAP_DSHOW)
    est = AprilTagCubeEstimator(quad_decimate=quad_decimate,
                                quad_sigma=quad_sigma)

    focal = 448
    while True:
        ok, frame = cap.read()
        if not ok:
            break

        poses, tags, _ = est.detect_cubes(frame, focal, use_thumb_offset=False)

        # I unfortunately decided to flip the frame vertically in the main tracking so test it properly I have to flip it here too ;(
        frame_flipped = cv2.flip(frame, 1) 
        hand_preds     = est(frame_flipped, focal)           # key-points

        h, w = frame.shape[:2]
        cam_p = (focal, focal, w*.5, h*.5)

        annotate_frame(frame, poses, tags, hand_preds, cam_p)

        cv2.imshow("AprilTag Cubes", frame)
        if cv2.waitKey(1) & 0xFF == 27:             # Esc quits
            break

    cap.release()
    cv2.destroyAllWindows()

# ─── CLI ──────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys
    cam_idx       = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    quad_decimate = float(sys.argv[2]) if len(sys.argv) > 2 else 1.0
    quad_sigma    = float(sys.argv[3]) if len(sys.argv) > 3 else 0.0
    main(cam_idx, quad_decimate, quad_sigma)