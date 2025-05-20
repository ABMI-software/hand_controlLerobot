import os
import xml.etree.ElementTree as ET
from xml.dom import minidom

ROBOT_MEASUREMENTS = {
    "koch": {
        "gripper": [0.239, -0.001, 0.024],
        "wrist": [0.209, 0, 0.024],
        "forearm": [0.108, 0, 0.02],
        "humerus": [0, 0, 0.036],
        "shoulder": [0, 0, 0],
        "base": [0, 0, -0.02],
    },
    "so100": {
        "gripper": [0.320, 0, 0.050],
        "wrist": [0.278, 0, 0.050],
        "forearm": [0.143, 0, 0.044],
        "humerus": [0.031, 0, 0.072],
        "shoulder": [0, 0, 0],
        "base": [0, 0, -0.02],
    },
    "moss": {
        "gripper": [0.246, 0.013, 0.111],
        "wrist": [0.245, 0.002, 0.064],
        "forearm": [0.122, 0, 0.064],
        "humerus": [0.001, 0.001, 0.063],
        "shoulder": [0, 0, 0],
        "base": [0, 0, -0.02],
    },
}

AXES = {
    "shoulder": [0, 0, -1],
    "humerus": [0, -1, 0],
    "forearm": [0, 1, 0],
    "wrist": [0, 1, 0],
    "gripper": [1, 0, 0],
}

RPYS = {
    "gripper": [-1.5708, 0, 0],
}

def make_joint(name, parent, child, origin, axis):
    joint = ET.Element("joint", name=f"{name}_joint", type="revolute")
    ET.SubElement(joint, "parent", link=f"{parent}_link")
    ET.SubElement(joint, "child", link=f"{child}_link")
    ET.SubElement(joint, "origin", xyz="{} {} {}".format(*origin), rpy="{} {} {}".format(*RPYS.get(name, [0, 0, 0])))
    ET.SubElement(joint, "axis", xyz="{} {} {}".format(*axis))
    ET.SubElement(joint, "limit", lower="-3.1416", upper="3.1416", effort="1", velocity="1")
    return joint

def make_urdf(name, measurements):
    robot = ET.Element("robot", name=f"{name}_minimal")

    link_order = ["base", "shoulder", "humerus", "forearm", "wrist", "gripper"]
    for link in link_order:
        ET.SubElement(robot, "link", name=f"{link}_link")

    for i in range(1, len(link_order)):
        parent = link_order[i - 1]
        child = link_order[i]

        origin = [round(child_i - parent_i, 6) for child_i, parent_i in zip(measurements[child], measurements[parent])]
        axis = AXES.get(child, [0, 0, 1])
        joint = make_joint(child, parent, child, origin, axis)
        robot.append(joint)

    return robot

def save_urdf_to_file(robot_elem, file_path):
    xml_str = ET.tostring(robot_elem, encoding='unicode')
    pretty_xml = minidom.parseString(xml_str).toprettyxml(indent="    ")
    with open(file_path, "w") as f:
        f.write(pretty_xml)

# Generate and save URDFs for all robot configurations
script_dir = os.path.dirname(os.path.abspath(__file__))
ROBOT_MEASUREMENTS["so101"] = ROBOT_MEASUREMENTS["so100"]

for robot_name, measurements in ROBOT_MEASUREMENTS.items():
    urdf = make_urdf(robot_name, measurements)
    output_path = os.path.join(script_dir, f"{robot_name}.urdf")
    save_urdf_to_file(urdf, output_path)