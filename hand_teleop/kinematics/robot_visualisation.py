import os
import time
import numpy as np
import xml.etree.ElementTree as ET
import matplotlib.pyplot as plt

class RobotVisualisation:
    def __init__(self, kinematics, urdf_name: str):
        self.kinematics = kinematics
        self.urdf_path = self._resolve_urdf_path(urdf_name)
        self.link_names, self.link_pairs = self._parse_urdf(self.urdf_path)

    def _resolve_urdf_path(self, name):
        base_dir = "C:\\github_personal\\hand-teleop\\hand_teleop\\kinematics\\urdf"
        if not name.endswith(".urdf"):
            name += ".urdf"
        return name if os.path.exists(name) else os.path.join(base_dir, name)

    def _parse_urdf(self, path):
        root = ET.parse(path).getroot()
        links = [l.attrib['name'] for l in root.findall('link')]
        joints = [(j.find('parent').attrib['link'], j.find('child').attrib['link'])
                  for j in root.findall('joint')]
        return links, joints

    def draw(self, ax, q):
        poses = {name: self.kinematics.fk(q, name)[:3, 3] for name in self.link_names[:6]}
        ax.clear()
        ax.set_xlim([-0.3, 0.3])
        ax.set_ylim([-0.3, 0.3])
        ax.set_zlim([0, 0.6])
        for parent, child in self.link_pairs:
            if parent in poses and child in poses:
                ax.plot(*zip(poses[parent], poses[child]), c='k')
        for pos in poses.values():
            ax.scatter(*pos, s=30, c='r')

if __name__ == "__main__":
    from hand_teleop.kinematics.kinematics import RobotKinematics
    urdf_name = "so100"
    urdf_path = f"C:\\github_personal\\hand-teleop\\hand_teleop\\kinematics\\urdf\\{urdf_name}.urdf"
    kin = RobotKinematics(urdf_path)
    viz = RobotVisualisation(kin, urdf_name)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    q = np.zeros(5)

    for t in range(100):
        q = 0.5 * np.sin(np.linspace(0, np.pi * 2, 5) + 0.1 * t)
        viz.draw(ax, q)
        plt.pause(0.05)