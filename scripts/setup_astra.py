#!/usr/bin/env python3
"""
Astra camera setup and diagnostic tool.
"""
import subprocess
import time
import sys
import os
from pathlib import Path

def run_cmd(cmd, check=True):
    """Run a command and return its output."""
    try:
        result = subprocess.run(cmd, shell=True, check=check, 
                              text=True, capture_output=True)
        return result.stdout.strip()
    except subprocess.CalledProcessError as e:
        print(f"Command failed: {e}")
        if check:
            sys.exit(1)
        return None

def main():
    print("Astra Camera Setup and Diagnostics")
    print("==================================")
    
    # Check for root privileges
    if os.geteuid() != 0:
        print("This script must be run as root (sudo).")
        sys.exit(1)
    
    # 1. Kill existing processes
    print("\n1. Cleaning up existing processes...")
    run_cmd("pkill -f 'openni|orbbec|oni'", check=False)
    run_cmd("rm -f /dev/shm/oni_*")
    
    # 2. Check USB device
    print("\n2. Checking USB device...")
    usb_devices = run_cmd("lsusb | grep -i 'astra\|orbbec'")
    if not usb_devices:
        print("! Astra camera not found. Please check the USB connection.")
        sys.exit(1)
    print(f"✓ Found device: {usb_devices}")
    
    # 3. Check OpenNI installation
    print("\n3. Checking OpenNI installation...")
    openni_libs = run_cmd("ldconfig -p | grep -i openni")
    if not openni_libs:
        print("! OpenNI libraries not found. Please install OpenNI2.")
        sys.exit(1)
    print("✓ OpenNI libraries found")
    
    # 4. Setup udev rules
    print("\n4. Setting up udev rules...")
    rule_content = 'SUBSYSTEM=="usb", ATTR{idVendor}=="2bc5", ATTR{idProduct}=="0402", MODE="0666"'
    rule_file = Path("/etc/udev/rules.d/99-orbbec-astra.rules")
    rule_file.write_text(rule_content)
    run_cmd("udevadm control --reload-rules")
    run_cmd("udevadm trigger")
    print("✓ udev rules updated")
    
    # 5. Reset USB device
    print("\n5. Resetting USB device...")
    run_cmd("usb_modeswitch -v 2bc5 -p 0402 -R", check=False)
    time.sleep(2)
    
    print("\nSetup complete! You can now run:")
    print("ONI_LOG_SEVERITY=verbose oni_grabber --no-ir")
    
    # Optionally start the camera service
    if "--start" in sys.argv:
        print("\nStarting camera service...")
        os.environ["ONI_LOG_SEVERITY"] = "verbose"
        os.execvp("oni_grabber", ["oni_grabber", "--no-ir"])

if __name__ == "__main__":
    main()