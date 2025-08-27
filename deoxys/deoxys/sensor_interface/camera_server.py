import pyrealsense2 as rs
import time
import cv2
import zmq
import time
import struct
from collections import deque
import numpy as np

from deoxys.sensor_interface.sensors.realsense import RealSenseCamera
from deoxys.sensor_interface.sensors.opencv_cam import OpenCVCamera
from deoxys.utils.cam_utils import load_camera_config, CameraInfo


class CameraServer:
    def __init__(self, config, port = 10001, Unit_Test = False):
        """
        config example1:
        {
            cam_host : 192.168.1.113
            cam_port : 10001

            cam_info:
                - cam_id: 0
                  cam_serial_num: '310222078614'
                  type: realsense
                  name: camera_0

                - cam_id: 1
                  cam_serial_num: '152122075567'
                  type: realsense
                  name: camera_1

                - cam_id: 2
                  cam_serial_num: '243322072209'
                  type: realsense
                  name: camera_2

                - cam_id: 3
                  cam_serial_num: '317222073629'
                  type: realsense
                  name: camera_3

            cam_config:
                realsense:
                  width: 640
                  height: 480
                  fps: 30
                  processing_preset: 1
                  depth: false

                opencv:
                  width: 640
                  height: 480
                  fps: 30
        }
        """
        
        self.cam_configs = config
        
        for cam_info in config['camera_infos']:
            # If cam_info is a dict, use ['camera_type'], if CameraInfo object, use .camera_type
            cam_type = cam_info['camera_type'] if isinstance(cam_info, dict) else cam_info.camera_type
            if cam_type == "realsense":
                ctx = rs.context()
                devices = ctx.query_devices()
                for dev in devices:
                    dev.hardware_reset()
                print("Waiting for hardware reset on cameras for 15 seconds...")
                time.sleep(5)
                break
        
        print(config)
        # self.fps = config.get('fps', 30)
        self.camera_infos = config.get('camera_infos', [])
        self.port = config.get('camera_port', 10001)
        self.Unit_Test = Unit_Test

        # Initialize all cameras
        self.cameras = []
        for cam_info in self.camera_infos:
            # If cam_info is a dict, use dict access, else attribute access
            if isinstance(cam_info, dict):
                cam_type = cam_info.get('camera_type', 'opencv')
                img_shape = cam_info.get('image_shape', [cam_info.get('height', 480), cam_info.get('width', 640)])
                cam_id = cam_info.get('camera_id', 0)
                fps = cam_info.get('fps', 30)
                serial_number = cam_info.get('camera_serial_num', None)
            else:
                cam_type = getattr(cam_info, 'camera_type', 'opencv')
                img_shape = getattr(cam_info, 'image_shape', [getattr(cam_info.cfg, 'height', 480), getattr(cam_info.cfg, 'width', 640)])
                cam_id = getattr(cam_info, 'camera_id', 0)
                fps = getattr(cam_info.cfg, 'fps', 30) if hasattr(cam_info, 'cfg') else 30
                serial_number = getattr(cam_info, 'camera_serial_num', None)
            if cam_type == 'opencv':
                camera = OpenCVCamera(device_id=cam_id, img_shape=img_shape, fps=fps)
            elif cam_type == 'realsense':
                camera = RealSenseCamera(img_shape=img_shape, fps=fps, serial_number=serial_number)
            else:
                print(f"[Image Server] Unsupported camera_type: {cam_type}")
                continue
            self.cameras.append(camera)

        # Set ZeroMQ context and socket
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.PUB)
        self.socket.bind(f"tcp://*:{self.port}")

        if self.Unit_Test:
            self._init_performance_metrics()

        for cam in self.cameras:
            if isinstance(cam, OpenCVCamera):
                print(f"[Image Server] Camera {cam.id} resolution: {cam.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)} x {cam.cap.get(cv2.CAP_PROP_FRAME_WIDTH)}")
            elif isinstance(cam, RealSenseCamera):
                print(f"[Image Server] Camera {cam.serial_number} resolution: {cam.img_shape[0]} x {cam.img_shape[1]}")
            else:
                print("[Image Server] Unknown camera type.")

        print("[Image Server] Image server has started, waiting for client connections...")

    def _init_performance_metrics(self):
        self.frame_count = 0  # Total frames sent
        self.time_window = 1.0  # Time window for FPS calculation (in seconds)
        self.frame_times = deque()  # Timestamps of frames sent within the time window
        self.start_time = time.time()  # Start time of the streaming

    def _update_performance_metrics(self, current_time):
        # Add current time to frame times deque
        self.frame_times.append(current_time)
        # Remove timestamps outside the time window
        while self.frame_times and self.frame_times[0] < current_time - self.time_window:
            self.frame_times.popleft()
        # Increment frame count
        self.frame_count += 1

    def _print_performance_metrics(self, current_time):
        if self.frame_count % 30 == 0:
            elapsed_time = current_time - self.start_time
            real_time_fps = len(self.frame_times) / self.time_window
            print(f"[Image Server] Real-time FPS: {real_time_fps:.2f}, Total frames sent: {self.frame_count}, Elapsed time: {elapsed_time:.2f} sec")

    def _close(self):
        for cam in self.cameras:
            cam.release()
        self.socket.close()
        self.context.term()
        print("[Image Server] The server has been closed.")

    def send_process(self):
        try:
            while True:
                # Grab frames from all cameras and concatenate horizontally
                frames = []
                for cam in self.cameras:
                    if isinstance(cam, OpenCVCamera):
                        color_image = cam.get_frame()
                        if color_image is None:
                            print("[Image Server] Camera frame read error (OpenCV).")
                            break
                        frames.append(color_image)
                    elif isinstance(cam, RealSenseCamera):
                        color_image, _ = cam.get_frame()
                        if color_image is None:
                            print("[Image Server] Camera frame read error (RealSense).")
                            break
                        frames.append(color_image)
                    else:
                        print("[Image Server] Unknown camera type.")
                        break

                if len(frames) != len(self.cameras):
                    continue  # Skip this iteration if any camera failed

                # Concatenate all frames horizontally
                full_color = cv2.hconcat(frames)

                ret, buffer = cv2.imencode('.jpg', full_color)
                if not ret:
                    print("[Image Server] Frame imencode is failed.")
                    continue

                jpg_bytes = buffer.tobytes()
          
                if self.Unit_Test:
                    timestamp = time.time()
                    frame_id = self.frame_count
                    header = struct.pack('dI', timestamp, frame_id)  # 8-byte double, 4-byte unsigned int
                    message = header + jpg_bytes
                else:
                    message = jpg_bytes

                self.socket.send(message)
 
                if self.Unit_Test:
                    current_time = time.time()
                    self._update_performance_metrics(current_time)
                    self._print_performance_metrics(current_time)

        except KeyboardInterrupt:
            print("[Image Server] Interrupted by user.")
        finally:
            self._close()


if __name__ == "__main__":
    
    config = load_camera_config()    
    server = CameraServer(config, Unit_Test=True)
    server.send_process()
    