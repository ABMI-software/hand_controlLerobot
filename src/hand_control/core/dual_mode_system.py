#!/usr/bin/env python3
"""
Main script for switching between object detection and teleoperation modes.
"""

import cv2
import numpy as np
import time
import subprocess
import os
import sys
import signal
import psutil
from pathlib import Path
from hand_control.cameras.camera_manager import CameraManager
from hand_control.detection.object_detector import ObjectDetector

class DualModeSystem:
    def __init__(self, model_path="yolov8n.pt", camera_id=0):
        self.running = True
        self.detections_queue = []  # Store recent detections
        self.mode = "detection"  # Current operation mode
        
        # Initialize object detection
        print("\n1. Setting up object detection...")
        try:
            self.cam_manager = CameraManager()
            
            # Try to find the ArduCam
            print("Setting up ArduCam for object detection...")
            camera_found = False
            for cam_id in range(4):
                try:
                    print(f"Trying camera at /dev/video{cam_id}...")
                    self.cam_manager.add_camera(
                        name="arducam",
                        camera_id=cam_id,
                        width=1280,
                        height=720,
                        fps=30
                    )
                    # Try to get a test frame
                    self.cam_manager.start()
                    frame, success = self.cam_manager.get_frame("arducam")
                    if success and frame is not None:
                        camera_found = True
                        print(f"✓ ArduCam found and working at /dev/video{cam_id}")
                        break
                    self.cam_manager.stop()
                except Exception as e:
                    print(f"Camera at /dev/video{cam_id} not available: {e}")
                    continue
                    
            if not camera_found:
                raise RuntimeError("ArduCam not found")
                
            # Download and load YOLOv8 model with COCO classes
            if not Path(model_path).exists():
                print(f"Downloading YOLOv8 model to {model_path}")
                import torch
                torch.hub.download_url_to_file(
                    'https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n.pt',
                    model_path
                )
            
            if not Path(model_path).exists():
                print(f"Downloading YOLOv8 model to {model_path}")
                import torch
                torch.hub.download_url_to_file(
                    'https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n.pt',
                    model_path
                )
            
            self.detector = ObjectDetector(model_path, conf_threshold=0.5)
            print("✓ Object detector initialized")
            
        except Exception as e:
            print(f"✗ Failed to initialize object detection: {e}")
            raise
            
        # Set up signal handler for clean shutdown
        signal.signal(signal.SIGINT, self._signal_handler)
        
    def _cleanup_resources(self, wait_timeout=3):
        """Clean up all camera and window resources with better error handling and timeouts.
        
        Args:
            wait_timeout (int): Maximum time to wait for processes to terminate gracefully
        """
        processes_to_terminate = []
        
        # First stop our camera manager
        if hasattr(self, 'cam_manager'):
            try:
                self.cam_manager.stop()
                print("✓ Camera manager stopped")
            except Exception as e:
                print(f"! Warning: Failed to stop camera manager: {e}")
        
        # Find all processes to terminate
        for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
            try:
                cmdline = ' '.join(proc.info['cmdline'] or [])
                if ('oni' in proc.info['name'].lower() or 
                    'hand_teleop_local.py' in cmdline or
                    'openni' in cmdline.lower()):
                    processes_to_terminate.append(proc)
            except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
                continue
        
        # Terminate processes gracefully first
        for proc in processes_to_terminate:
            try:
                print(f"Terminating process: {proc.name()} (PID: {proc.pid})")
                proc.terminate()
            except psutil.NoSuchProcess:
                continue
        
        # Wait for processes to terminate
        _, alive = psutil.wait_procs(processes_to_terminate, timeout=wait_timeout)
        
        # Force kill any remaining processes
        for proc in alive:
            try:
                print(f"Force killing process: {proc.name()} (PID: {proc.pid})")
                proc.kill()
            except psutil.NoSuchProcess:
                continue
        
        # Close all OpenCV windows
        cv2.destroyAllWindows()
        
        # Wait a short moment for all resources to be fully released
        time.sleep(0.5)
        
        # Release system resources
        import gc
        gc.collect()

    def _signal_handler(self, signum, frame):
        """Handle Ctrl+C gracefully"""
        print("\nShutting down...")
        self.running = False
        self._cleanup_resources()
        
    def run_object_detection(self):
        """Run object detection mode using Arducam with object classification"""
        print("\nStarting object detection mode...")
        print("Press 't' to switch to teleoperation mode")
        print("Press 'q' to quit")
        
        # Create windows
        cv2.namedWindow("Object Detection", cv2.WINDOW_NORMAL)
        cv2.setWindowProperty("Object Detection", cv2.WND_PROP_TOPMOST, 1)
        
        frame_timeout = 0
        last_detection_time = time.time()
        
        while self.running and self.mode == "detection":
            # Get frame from Arducam with timeout
            frame, success = self.cam_manager.get_frame("arducam")
            if not success or frame is None:
                frame_timeout += 1
                if frame_timeout > 10:
                    print("No frames received from camera. Checking camera status...")
                    self.cam_manager.stop()
                    time.sleep(0.5)
                    try:
                        self.cam_manager.start()
                        frame_timeout = 0
                    except Exception as e:
                        print(f"Failed to restart camera: {e}")
                        self.running = False
                        break
                time.sleep(0.1)
                continue
                
            frame_timeout = 0  # Reset timeout counter on successful frame
            
            # Run detection
            detections = self.detector.detect(frame)
            
            # Process and display detections
            viz_frame = frame.copy()
            if detections:
                current_time = time.time()
                # Only update detections every 0.5 seconds
                if current_time - last_detection_time >= 0.5:
                    self.detections_queue = detections
                    last_detection_time = current_time
                
                # Draw detections with labels
                for det in detections:
                    box = det['bbox']
                    conf = det['confidence']
                    cls = det['class']
                    label = f"{self.detector.get_class_name(cls)}: {conf:.2f}"
                    
                    # Draw box
                    cv2.rectangle(viz_frame, 
                                (int(box[0]), int(box[1])),
                                (int(box[2]), int(box[3])),
                                (0, 255, 0), 2)
                    
                    # Draw label with background
                    label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
                    cv2.rectangle(viz_frame,
                                (int(box[0]), int(box[1]) - label_size[1] - 10),
                                (int(box[0]) + label_size[0], int(box[1])),
                                (0, 255, 0), -1)
                    cv2.putText(viz_frame, label,
                              (int(box[0]), int(box[1]) - 5),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
                
            # Run detection
            detections = self.detector.detect(frame)
            viz_frame = self.detector.visualize(frame, detections)
            
            # Add mode info
            cv2.putText(viz_frame, "Object Detection Mode", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(viz_frame, "Press 't' for teleoperation", (10, 70),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            try:
                # Show frame
                cv2.imshow("Main View", viz_frame)
                
                # Handle key presses - ensure window focus and proper event handling
                key = cv2.waitKey(1)  # Shorter wait time for responsiveness
                if key == -1:  # No key pressed
                    continue
                    
                key = key & 0xFF  # Get the actual key value
                if key == ord('q'):
                    print("Quit command received")
                    self.running = False
                    break
                elif key == ord('t'):
                    print("Switching to teleoperation mode")
                    return "teleoperation"
                elif key != 255:  # If any other key was pressed
                    print(f"Key pressed: {chr(key) if 32 <= key <= 126 else key}")
                    
            except cv2.error as e:
                print(f"OpenCV error: {e}")
                # Try to recover
                cv2.destroyAllWindows()
                time.sleep(0.1)
                cv2.namedWindow("Main View", cv2.WINDOW_NORMAL)
                continue
                
        return "quit"
        
    def run_teleoperation(self):
        """Run teleoperation mode using both cameras: Astra for hand tracking and ArduCam for detection"""
        print("\nStarting dual-camera teleoperation mode...")
        print("Press 'd' to switch to detection-only mode")
        print("Press 'q' to quit")
        
        # First check if Astra camera is available
        print("\nChecking Astra camera status...")
        try:
            import subprocess
            result = subprocess.run(['lsusb'], capture_output=True, text=True)
            if '2bc5:0402' not in result.stdout:
                print("! Error: Astra camera not detected. Please check USB connection.")
                return "detection"
        except Exception as e:
            print(f"! Error checking USB devices: {e}")
        
        # Check OpenNI shared memory
        if not all(Path(f"/dev/shm/{f}").exists() for f in ["oni_color.rgb", "oni_info.txt", "oni_tick.txt"]):
            print("\nStarting OpenNI camera service...")
            try:
                subprocess.Popen(["oni_grabber", "--no-ir"], 
                               stdout=subprocess.PIPE, 
                               stderr=subprocess.PIPE)
                time.sleep(2)  # Wait for service to start
            except Exception as e:
                print(f"! Failed to start OpenNI service: {e}")
                print("Please make sure OpenNI is properly installed:")
                print("1. Check if OpenNI is installed: dpkg -l | grep openni")
                print("2. If not installed: sudo apt install openni-utils libopenni-dev")
                return "detection"
        
        # Keep ArduCam running and add window for teleoperation
        cv2.namedWindow("Teleoperation", cv2.WINDOW_NORMAL)
        cv2.setWindowProperty("Teleoperation", cv2.WND_PROP_TOPMOST, 1)
        
        print("\nInitializing hand tracking...")
        
        # Start teleoperation process with better error handling
        script_path = Path(__file__).parent.parent.parent.parent / "scripts" / "hand_teleop_local.py"
        if not script_path.exists():
            print(f"! Error: Teleoperation script not found at {script_path}")
            return "detection"
            
        # Clean up any existing OpenNI processes
        subprocess.run(['pkill', '-f', 'oni_grabber'], capture_output=True)
        for f in Path('/dev/shm').glob('oni_*'):
            try:
                f.unlink()
            except Exception:
                pass
        
        time.sleep(1)  # Wait for cleanup
            
        cmd = [
            "python3", str(script_path),
            "--hand", "right",
            "--model", "wilor",
            "--cam-idx", "-1",  # OpenNI interface for Astra
            "--fps", "30",
            "--so101-enable",
            "--so101-port", "/dev/serial/by-id/usb-1a86_USB_Single_Serial_5AA9018150-if00",
            "--invert-z",
            "--raw",
            "--raw-min", "1700",
            "--raw-max", "3200",
            "--verbose",
            "--print-joints"
        ]
        
        # Start OpenNI service first
        try:
            oni_process = subprocess.Popen(["oni_grabber", "--no-ir"], 
                                         stdout=subprocess.PIPE, 
                                         stderr=subprocess.PIPE)
            time.sleep(2)  # Wait for service to start
        except Exception as e:
            print(f"! Failed to start OpenNI service: {e}")
            return "detection"
        
        process = None
        try:
            # Run in a new process group with higher priority
            process = subprocess.Popen(
                cmd,
                preexec_fn=lambda: (
                    os.setsid(),
                    os.nice(-10)  # Give higher priority to teleoperation
                ),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                bufsize=1,  # Line buffered
                universal_newlines=True  # Text mode
            )
            
            print("✓ Teleoperation process started successfully")
            
            # Monitor both cameras and process
            last_alive_check = time.time()
            last_detection_time = time.time()
            
            while self.running:
                # Check teleoperation process health
                current_time = time.time()
                if current_time - last_alive_check >= 2:
                    if process.poll() is not None:
                        print("! Teleoperation process ended unexpectedly")
                        _, stderr = process.communicate(timeout=1)
                        if stderr:
                            print("Error output:", stderr)
                        break
                    last_alive_check = current_time
                    
                # Continue running object detection with ArduCam
                frame, success = self.cam_manager.get_frame("arducam")
                if success and frame is not None:
                    # Run detection
                    detections = self.detector.detect(frame)
                    viz_frame = frame.copy()
                    
                    # Process and display detections
                    if detections:
                        current_time = time.time()
                        if current_time - last_detection_time >= 0.5:
                            self.detections_queue = detections
                            last_detection_time = current_time
                            print("\nDetected objects:")
                            for det in detections:
                                cls_name = self.detector.get_class_name(det['class'])
                                print(f"- {cls_name} ({det['confidence']:.2f})")
                        
                        # Draw detections
                        for det in detections:
                            box = det['bbox']
                            label = f"{self.detector.get_class_name(det['class'])}: {det['confidence']:.2f}"
                            cv2.rectangle(viz_frame, 
                                        (int(box[0]), int(box[1])),
                                        (int(box[2]), int(box[3])),
                                        (0, 255, 0), 2)
                            cv2.putText(viz_frame, label,
                                      (int(box[0]), int(box[1]) - 10),
                                      cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                    
                    # Show object detection window
                    cv2.imshow("Object Detection", viz_frame)
                
                # Handle keyboard input (non-blocking)
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    self.running = False
                    break
                elif key == ord('d'):
                    return "detection"
                
                # Shorter sleep to be more responsive while keeping CPU usage reasonable
                time.sleep(0.005)
                
        except Exception as e:
            print(f"! Error in teleoperation: {e}")
            return "detection"  # Try to recover by going back to detection mode
            
        finally:
            # Ensure clean process termination
            if process:
                try:
                    # Try graceful termination first
                    os.killpg(os.getpgid(process.pid), signal.SIGTERM)
                    try:
                        process.wait(timeout=3)  # Wait up to 3 seconds for graceful shutdown
                    except subprocess.TimeoutExpired:
                        # Force kill if graceful shutdown takes too long
                        os.killpg(os.getpgid(process.pid), signal.SIGKILL)
                except (ProcessLookupError, OSError) as e:
                    print(f"! Note: Process already terminated: {e}")
            
            # Clean up and prepare for detection mode
            self._cleanup_resources()
            
            # Restart Arducam for detection mode with retry logic
            max_retries = 3
            for attempt in range(max_retries):
                try:
                    self.cam_manager.start()
                    print("✓ Successfully restarted camera for detection mode")
                    break
                except Exception as e:
                    if attempt < max_retries - 1:
                        print(f"! Warning: Failed to restart camera (attempt {attempt + 1}): {e}")
                        time.sleep(1)  # Wait before retry
                    else:
                        print(f"! Error: Failed to restart camera after {max_retries} attempts")
            
        return "quit" if not self.running else "detection"
        
    def run(self):
        """Main run loop switching between modes"""
        mode = "detection"  # Start with object detection
        
        while self.running:
            if mode == "detection":
                mode = self.run_object_detection()
            elif mode == "teleoperation":
                mode = self.run_teleoperation()
            else:  # quit
                break
                
        # Cleanup
        self._cleanup_resources()

def main():
    try:
        # Create and run the system
        system = DualModeSystem()
        
        # Clean up any existing processes before starting
        system._cleanup_resources()
        
        # Run the system
        system.run()
    except KeyboardInterrupt:
        print("\nShutting down...")
    except Exception as e:
        print(f"Error: {e}")
    finally:
        # Ensure all resources are cleaned up
        if 'system' in locals():
            system._cleanup_resources()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()