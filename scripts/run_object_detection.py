#!/usr/bin/env python3
"""
Entry point script for object detection mode.
"""
from hand_control.core.dual_mode_system import DualModeSystem

def main():
    """Run object detection mode."""
    system = DualModeSystem()
    system.run_object_detection()

if __name__ == "__main__":
    main()