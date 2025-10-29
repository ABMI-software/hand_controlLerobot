from __future__ import annotations

from pathlib import Path
import numpy as np
import xml.etree.ElementTree as ET
import matplotlib.pyplot as plt


def _localname(tag: str) -> str:
    """Return the XML localname (strip '{ns}')."""
    return tag.split('}', 1)[-1] if '}' in tag else tag


class RobotVisualisation:
    """
    URDF-based visualiser:
      - parses links and parent→child joint pairs from a URDF
      - calls kinematics.fk(q_rad, link_name) to get each link pose
      - draws a 3D skeleton and a small gripper-angle inset

    Expected joint vector: [j1..j5, gripper_open] in DEGREES.
    Only the first 5 go into FK (as radians).
    """

    def __init__(self, kinematics, urdf_name_or_path: str):
        self.kinematics = kinematics
        self.urdf_path = self._resolve_urdf_path(urdf_name_or_path)
        self.link_names, self.link_pairs = self._parse_urdf(self.urdf_path)

        # Inset state
        self._inset_ax = None
        self._arc_line = None
        self._angle_line = None
        self._angle_text = None

    # ---------------------------- helpers ---------------------------- #

    def _resolve_urdf_path(self, name_or_path: str) -> str:
        """
        Accept either:
          - a direct path to a .urdf file (absolute or relative), or
          - a bare model name like 'so100' which will be resolved to
            <this_module_dir>/urdf/so100.urdf
        """
        p = Path(name_or_path)

        # Direct path case
        if p.suffix == ".urdf" and p.exists():
            return str(p)

        # Bare name → resolve to the 'urdf' folder next to this file
        if p.suffix == "":
            candidate = Path(__file__).resolve().parent / "urdf" / f"{name_or_path}.urdf"
            if candidate.exists():
                return str(candidate)

        # Relative path missing suffix but exists with .urdf
        if p.suffix == "" and p.with_suffix(".urdf").exists():
            return str(p.with_suffix(".urdf"))

        raise FileNotFoundError(
            f"URDF not found for '{name_or_path}'. "
            f"Tried '{p}' and '{Path(__file__).resolve().parent / 'urdf' / (str(name_or_path) + '.urdf')}'."
        )

    def _parse_urdf(self, path: str):
        """
        Parse the URDF and return:
          - link_names: list[str]
          - link_pairs: list[(parent_link_name, child_link_name)]
        Namespace-safe (handles <robot xmlns="...">).
        """
        root = ET.parse(path).getroot()

        links = []
        for node in root.iter():
            if _localname(node.tag) == "link":
                name = node.attrib.get("name")
                if name:
                    links.append(name)

        joints = []
        for j in root.iter():
            if _localname(j.tag) == "joint":
                parent = None
                child = None
                for sub in j:
                    ln = _localname(sub.tag)
                    if ln == "parent":
                        parent = sub.attrib.get("link")
                    elif ln == "child":
                        child = sub.attrib.get("link")
                if parent and child:
                    joints.append((parent, child))

        return links, joints

    # ----------------------------- draw ------------------------------ #

    def draw(self, ax, q):
        """
        Visualize the robot in 3D using joint configuration.

        Expects a 6-element joint array in DEGREES:
        [j1, j2, j3, j4, j5, gripper_open]
        """
        q = np.asarray(q).flatten()
        assert q.size >= 6, "Expected at least 6 joints [j1..j5, gripper_open] (degrees)"

        # Convert first 5 joint angles to radians for FK
        q_rad = np.radians(q[:5])

        # FK for all links we know about (skip if FK can't provide a link)
        poses = {}
        for name in self.link_names:
            try:
                T = self.kinematics.fk(q_rad, name)
                poses[name] = T[:3, 3]
            except Exception:
                # Some kinematics implementations may not expose all links
                continue

        # --- 3D skeleton ------------------------------------------------------
        ax.clear()
        ax.set_xlim([-0.3, 0.3])
        ax.set_ylim([-0.3, 0.3])
        ax.set_zlim([0, 0.6])
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        ax.set_title(f"URDF: {Path(self.urdf_path).name}")

        # draw edges
        for parent, child in self.link_pairs:
            if parent in poses and child in poses:
                a = poses[parent]
                b = poses[child]
                ax.plot([a[0], b[0]], [a[1], b[1]], [a[2], b[2]], c="k")

        # draw nodes
        for pos in poses.values():
            ax.scatter(pos[0], pos[1], pos[2], s=30, c="r")

        # --- gripper angle inset ---------------------------------------------
        fig = ax.get_figure()
        if self._inset_ax is None:
            self._inset_ax = fig.add_axes((0.78, 0.78, 0.2, 0.2))
            self._inset_ax.set_aspect("equal")
            self._inset_ax.axis("off")
            self._inset_ax.add_artist(plt.Circle((0, 0), 1, fill=False))
            self._inset_ax.plot([0, 1], [0, 0], color="gray", linestyle=":")
            (self._arc_line,) = self._inset_ax.plot([], [], ls="--")
            (self._angle_line,) = self._inset_ax.plot([], [], lw=2)
            self._angle_text = self._inset_ax.text(0, -1.4, "", ha="center", fontsize=9)
            self._inset_ax.set_xlim(-1.1, 1.1)
            self._inset_ax.set_ylim(-1.3, 1.1)

        theta = np.radians(q[5])  # gripper angle in radians (input degrees)
        arc = np.linspace(0, theta, 64)
        self._arc_line.set_data(np.cos(arc), np.sin(arc))
        self._angle_line.set_data([0, np.cos(theta)], [0, np.sin(theta)])
        self._angle_text.set_text(f"{q[5]:.1f}°")
