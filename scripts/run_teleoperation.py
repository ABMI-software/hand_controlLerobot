#!/usr/bin/env python3
"""
Entry point script for teleoperation mode.
"""
from hand_control.core.dual_mode_system import DualModeSystem

def main():
    """Run teleoperation mode."""
    system = DualModeSystem()
    system.run_teleoperation()

if __name__ == "__main__":
    main()