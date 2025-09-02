import pyrealsense2 as rs
import time
import cv2
import zmq
import time
import struct
from collections import deque
import numpy as np
import os
import json
from deoxys.sensor_interface.sensors.realsense_v1 import RealSenseCamera_V1
from deoxys.utils.cam_utils import load_camera_config
from deoxys.sensor_interface.sensors.opencv_cam import OpenCVCamera_V1

class CameraServer:
    def __init__(self, config, port = 10001, Unit_Test = False):
        """
        cam_host : 192.168.1.113
        cam_port : 10001

        cam_infos:
        - cam_id: 0
        cam_serial_num: '310222078614'
        type: realsense
        name: camera_0
        cam_config:
            width: 640
            height: 480
            fps: 30
            processing_preset: 1
            color_sensor:
            enable: true
            auto_exposure: true
            auto_white_balance: true
            exposure: 166               # 曝光值 (1-10000)
            gain: 64                    # 增益值 (0-128)
            white_balance: 4600         # 白平衡值 (2800-6500)
            brightness: 0               # 亮度 (-64 to 64)
            contrast: 50                # 对比度 (0-100)
            saturation: 64              # 饱和度 (0-100)
            depth_sensor: 
            enable: false
            ir_sensors: 
            enable: false
            auto_exposure: true
            disable_emitter: false      # 是否关闭红外投射器
            exposure: 8500              # 曝光值 (1-200000)
            gain: 16                    # 增益值 (16-248)
            laser_power: 150            # 激光功率 (0-360)

        - cam_id: 1
        cam_serial_num: '347622076290'
        type: realsense
        name: camera_1
        cam_config:
            width: 640
            height: 480
            fps: 30
            processing_preset: 1
            color_sensor:
            enable: true
            auto_exposure: true
            auto_white_balance: true
            exposure: 166               # 曝光值 (1-10000)
            gain: 64                    # 增益值 (0-128)
            white_balance: 4600         # 白平衡值 (2800-6500)
            brightness: 0               # 亮度 (-64 to 64)
            contrast: 50                # 对比度 (0-100)
            saturation: 64              # 饱和度 (0-100)
            depth_sensor: 
            enable: false
            ir_sensors: 
            enable: false
            auto_exposure: true
            disable_emitter: false      # 是否关闭红外投射器
            exposure: 8500              # 曝光值 (1-200000)
            gain: 16                    # 增益值 (16-248)
            laser_power: 150            # 激光功率 (0-360)

        - cam_id: 2
        cam_serial_num: '405622076349'
        type: realsense
        name: camera_2
        cam_config:
            width: 640
            height: 480
            fps: 30
            processing_preset: 1
            color_sensor:
            enable: true
            auto_exposure: true
            auto_white_balance: true
            exposure: 166               # 曝光值 (1-10000)
            gain: 64                    # 增益值 (0-128)
            white_balance: 4600         # 白平衡值 (2800-6500)
            brightness: 0               # 亮度 (-64 to 64)
            contrast: 50                # 对比度 (0-100)
            saturation: 64              # 饱和度 (0-100)
            depth_sensor: 
            enable: false               # Enable depth stream for this camera
            ir_sensors: 
            enable: false
            auto_exposure: true
            disable_emitter: false      # 是否关闭红外投射器
            exposure: 8500              # 曝光值 (1-200000)
            gain: 16                    # 增益值 (16-248)
            laser_power: 150            # 激光功率 (0-360)

        - cam_id: 3
        cam_serial_num: '243322072209'
        type: realsense
        name: camera_3
        cam_config:
            width: 640
            height: 480
            fps: 30
            processing_preset: 1
            color_sensor:
            enable: true
            auto_exposure: true
            auto_white_balance: true
            exposure: 166               # 曝光值 (1-10000)
            gain: 64                    # 增益值 (0-128)
            white_balance: 4600         # 白平衡值 (2800-6500)
            brightness: 0               # 亮度 (-64 to 64)
            contrast: 50                # 对比度 (0-100)
            saturation: 64              # 饱和度 (0-100)
            depth_sensor: 
            enable: false
            ir_sensors: 
            enable: false
            auto_exposure: true
            disable_emitter: false      # 是否关闭红外投射器
            exposure: 8500              # 曝光值 (1-200000)
            gain: 16                    # 增益值 (16-248)
            laser_power: 150            # 激光功率 (0-360)

        - cam_id: 4
        cam_serial_num: '405622072676'
        type: realsense
        name: camera_4
        cam_config:
            width: 640
            height: 480
            fps: 30
            processing_preset: 1
            color_sensor:
            enable: true
            auto_exposure: true
            auto_white_balance: true
            exposure: 166               # 曝光值 (1-10000)
            gain: 64                    # 增益值 (0-128)
            white_balance: 4600         # 白平衡值 (2800-6500)
            brightness: 0               # 亮度 (-64 to 64)
            contrast: 50                # 对比度 (0-100)
            saturation: 64              # 饱和度 (0-100)
            depth_sensor: 
            enable: false               # Enable depth stream
            ir_sensors: 
            enable: true               # Enable IR streams
            auto_exposure: true
            disable_emitter: true       # 是否关闭红外投射器
            exposure: 8500              # 曝光值 (1-200000)
            gain: 16                    # 增益值 (16-248)
            laser_power: 150            # 激光功率 (0-360)

        """
        
        self.cam_configs = config
        
        for cam_info in config['camera_infos']:
            # If cam_info is a dict, use ['camera_type'], if CameraInfo object, use .camera_type
            cam_type = cam_info['camera_type'] if isinstance(cam_info, dict) else cam_info.camera_type
            if "realsense" in cam_type:
                ctx = rs.context()
                devices = ctx.query_devices()
                for dev in devices:
                    dev.hardware_reset()
                print("Waiting for hardware reset on cameras for 15 seconds...")
                time.sleep(5)
                break
        
        print(config)
        self.camera_infos = config.get('camera_infos', [])
        self.port = config.get('camera_port', 10001)
        self.Unit_Test = Unit_Test

        # Initialize all cameras
        self.cameras = []
        for cam_info in self.camera_infos:
            # Get camera configuration from CameraInfo object
            cam_type = getattr(cam_info, 'camera_type', 'opencv')
            cam_id = getattr(cam_info, 'camera_id', 0)
            serial_number = getattr(cam_info, 'camera_serial_num', None)
            cam_name = getattr(cam_info, 'camera_name', None)
            cam_config = getattr(cam_info, 'cfg', {})

            if cam_type == 'opencv':
                camera = OpenCVCamera_V1(device_id=cam_id, config=cam_config)
            elif cam_type == 'realsense':
                camera = RealSenseCamera_V1(config=cam_config, serial_number=serial_number)
                # Initialize the camera
                if not camera.initialize():
                    print(f"[Image Server] Failed to initialize camera {cam_name} ({serial_number})")
                    continue
            else:
                print(f"[Image Server] Unsupported camera_type: {cam_type}")
                continue
            
            # Store camera with additional info for easier access
            camera.cam_name = cam_name
            camera.cam_id = cam_id
            camera.cam_type = cam_type
            self.cameras.append(camera)

        # Set ZeroMQ context and socket
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.PUB)
        self.socket.bind(f"tcp://*:{self.port}")

        if self.Unit_Test:
            self._init_performance_metrics()

        for cam in self.cameras:
            if isinstance(cam, OpenCVCamera_V1):
                print(f"[Image Server] Camera {cam.cam_name} (ID: {cam.cam_id}) resolution: {cam.height} x {cam.width}")
            elif isinstance(cam, RealSenseCamera_V1):
                print(f"[Image Server] Camera {cam.cam_name} (Serial: {cam.serial_number}) resolution: {cam.height} x {cam.width}")
                # Print enabled streams
                streams = []
                if cam.enable_color:
                    streams.append("color")
                if cam.enable_depth:
                    streams.append("depth")
                if cam.enable_ir:
                    streams.append("ir")
                print(f"                  Enabled streams: {', '.join(streams)}")
            else:
                print("[Image Server] Unknown camera type.")

        print("[Image Server] Image server has started, waiting for client connections...")

    def _encode_depth_image(self, depth_image):
        """Convert depth image to 8-bit for transmission"""
        if depth_image is None:
            return None
        
        # Normalize depth to 0-255 range
        depth_normalized = cv2.normalize(depth_image, None, 0, 255, cv2.NORM_MINMAX)
        depth_8bit = depth_normalized.astype(np.uint8)
        
        # Convert single channel to 3 channel for consistency
        depth_3channel = cv2.cvtColor(depth_8bit, cv2.COLOR_GRAY2BGR)
        return depth_3channel

    def _encode_ir_image(self, ir_image):
        """Convert IR image to 3-channel for transmission"""
        if ir_image is None:
            return None
        
        # Convert single channel to 3 channel
        ir_3channel = cv2.cvtColor(ir_image, cv2.COLOR_GRAY2BGR)
        return ir_3channel

    def _create_stream_layout(self, color_img, depth_img, ir_left_img, ir_right_img):
        """
        Create a combined image layout from multiple streams
        Layout: [Color | Depth | IR_Left | IR_Right] (horizontal concatenation)
        """
        valid_images = []
        stream_info = []
        
        if color_img is not None:
            valid_images.append(color_img)
            stream_info.append("color")
        
        if depth_img is not None:
            depth_encoded = self._encode_depth_image(depth_img)
            if depth_encoded is not None:
                valid_images.append(depth_encoded)
                stream_info.append("depth")
        
        if ir_left_img is not None:
            ir_left_encoded = self._encode_ir_image(ir_left_img)
            if ir_left_encoded is not None:
                valid_images.append(ir_left_encoded)
                stream_info.append("ir_left")
        
        if ir_right_img is not None:
            ir_right_encoded = self._encode_ir_image(ir_right_img)
            if ir_right_encoded is not None:
                valid_images.append(ir_right_encoded)
                stream_info.append("ir_right")
        
        if not valid_images:
            return None, []
        
        # Concatenate horizontally
        combined_image = cv2.hconcat(valid_images)
        return combined_image, stream_info

    def _create_frame_metadata(self, cameras_data):
        """
        Create metadata describing the frame structure
        cameras_data: list of (cam_id, cam_name, stream_info, image_width, image_height)
        """
        metadata = {
            "cameras": [],
            "total_width": 0,
            "total_height": 0
        }
        
        total_width = 0
        max_height = 0
        
        for cam_id, cam_name, stream_info, img_width, img_height in cameras_data:
            cam_metadata = {
                "cam_id": cam_id,
                "cam_name": cam_name,
                "streams": stream_info,
                "start_x": total_width,
                "width": img_width,
                "height": img_height,
                "stream_count": len(stream_info)
            }
            metadata["cameras"].append(cam_metadata)
            total_width += img_width
            max_height = max(max_height, img_height)
        
        metadata["total_width"] = total_width
        metadata["total_height"] = max_height
        
        return metadata

    def _init_performance_metrics(self):
        self.frame_count = 0  # Total frames sent
        self.time_window = 1.0  # Time window for FPS calculation (in seconds)
        self.frame_times = deque()  # Timestamps of frames sent within the time window
        self.start_time = time.time()  # Start time of the streaming
        
        # Enhanced performance tracking
        self.capture_times = deque()  # Frame capture timestamps
        self.processing_times = deque()  # Frame processing durations
        self.transmission_times = deque()  # Frame transmission timestamps
        self.frame_sizes = deque()  # Frame sizes in bytes

    def _update_performance_metrics(self, capture_time, processing_time, transmission_time, frame_size):
        # Add timestamps to deques
        self.capture_times.append(capture_time)
        self.processing_times.append(processing_time)
        self.transmission_times.append(transmission_time)
        self.frame_sizes.append(frame_size)
        self.frame_times.append(transmission_time)
        
        # Remove timestamps outside the time window
        current_time = transmission_time
        while self.frame_times and self.frame_times[0] < current_time - self.time_window:
            self.frame_times.popleft()
        while self.capture_times and self.capture_times[0] < current_time - self.time_window:
            self.capture_times.popleft()
        while self.processing_times and self.processing_times[0] < current_time - self.time_window:
            self.processing_times.popleft()
        while self.transmission_times and self.transmission_times[0] < current_time - self.time_window:
            self.transmission_times.popleft()
        while self.frame_sizes and len(self.frame_sizes) > len(self.frame_times):
            self.frame_sizes.popleft()
            
        # Increment frame count
        self.frame_count += 1

    def _print_performance_metrics(self, current_time):
        if self.frame_count % 30 == 0:
            elapsed_time = current_time - self.start_time
            real_time_fps = len(self.frame_times) / self.time_window
            
            # Calculate processing metrics
            if len(self.processing_times) >= 2:
                recent_processing_times = list(self.processing_times)[-min(30, len(self.processing_times)):]
                avg_processing_time = sum(recent_processing_times) / len(recent_processing_times)
                max_processing_time = max(recent_processing_times)
                min_processing_time = min(recent_processing_times)
            else:
                avg_processing_time = max_processing_time = min_processing_time = 0
            
            # Calculate frame size metrics
            if self.frame_sizes:
                recent_frame_sizes = list(self.frame_sizes)[-min(30, len(self.frame_sizes)):]
                avg_frame_size = sum(recent_frame_sizes) / len(recent_frame_sizes)
                max_frame_size = max(recent_frame_sizes)
                min_frame_size = min(recent_frame_sizes)
                total_data_sent = sum(self.frame_sizes) / (1024 * 1024)  # MB
            else:
                avg_frame_size = max_frame_size = min_frame_size = total_data_sent = 0
            
            # Calculate pipeline latency (capture to transmission)
            if len(self.capture_times) >= 1 and len(self.transmission_times) >= 1:
                recent_capture = list(self.capture_times)[-min(30, len(self.capture_times)):]
                recent_transmission = list(self.transmission_times)[-min(30, len(self.transmission_times)):]
                if len(recent_capture) == len(recent_transmission):
                    pipeline_latencies = [(t - c) * 1000 for c, t in zip(recent_capture, recent_transmission)]
                    avg_pipeline_latency = sum(pipeline_latencies) / len(pipeline_latencies)
                    max_pipeline_latency = max(pipeline_latencies)
                    min_pipeline_latency = min(pipeline_latencies)
                else:
                    avg_pipeline_latency = max_pipeline_latency = min_pipeline_latency = 0
            else:
                avg_pipeline_latency = max_pipeline_latency = min_pipeline_latency = 0
            
            print(f"[Image Server] === Performance Metrics ===")
            print(f"  FPS: {real_time_fps:.2f} | Frames: {self.frame_count} | Runtime: {elapsed_time:.1f}s")
            print(f"  Processing: Avg={avg_processing_time*1000:.1f}ms, Max={max_processing_time*1000:.1f}ms, Min={min_processing_time*1000:.1f}ms")
            print(f"  Pipeline Latency: Avg={avg_pipeline_latency:.1f}ms, Max={max_pipeline_latency:.1f}ms, Min={min_pipeline_latency:.1f}ms")
            print(f"  Frame Size: Avg={avg_frame_size/1024:.1f}KB, Max={max_frame_size/1024:.1f}KB, Min={min_frame_size/1024:.1f}KB")
            print(f"  Total Data Sent: {total_data_sent:.2f}MB")
            print(f"  =====================================")
            
            # Calculate bandwidth
            if elapsed_time > 0:
                bandwidth_mbps = (total_data_sent * 8) / elapsed_time  # Mbps
                print(f"  Bandwidth: {bandwidth_mbps:.2f} Mbps")

    def _close(self):
        for cam in self.cameras:
            if hasattr(cam, 'release'):
                cam.release()
        self.socket.close()
        self.context.term()
        print("[Image Server] The server has been closed.")

    def send_process(self):
        try:
            while True:
                # === FRAME CAPTURE PHASE ===
                capture_start_time = time.time()
                
                # Collect frames from all cameras and build combined image
                all_camera_images = []
                cameras_metadata = []
                
                for cam in sorted(self.cameras, key=lambda c: c.cam_id):
                    if isinstance(cam, OpenCVCamera_V1):
                        color_image = cam.get_frame()
                        if color_image is None:
                            print(f"[Image Server] Camera frame read error (OpenCV) - {cam.cam_name}")
                            break
                        
                        # For OpenCV cameras, only color stream
                        combined_image, stream_info = self._create_stream_layout(color_image, None, None, None)
                        
                    elif isinstance(cam, RealSenseCamera_V1):
                        # Get all available streams from RealSense camera
                        frame_data = cam.get_frame()
                        if frame_data is None:
                            print(f"[Image Server] Camera frame read error (RealSense) - {cam.cam_name}")
                            break
                        
                        color_img, depth_img, ir_left_img, ir_right_img = frame_data
                        combined_image, stream_info = self._create_stream_layout(color_img, depth_img, ir_left_img, ir_right_img)
                    
                    else:
                        print(f"[Image Server] Unknown camera type - {cam.cam_name}")
                        break
                    
                    if combined_image is None:
                        print(f"[Image Server] No valid streams from camera - {cam.cam_name}")
                        break
                    
                    all_camera_images.append(combined_image)
                    
                    # Store metadata for this camera
                    img_height, img_width = combined_image.shape[:2]
                    cameras_metadata.append((cam.cam_id, cam.cam_name, stream_info, img_width, img_height))

                if len(all_camera_images) != len(self.cameras):
                    continue  # Skip this iteration if any camera failed

                capture_end_time = time.time()

                # === FRAME PROCESSING PHASE ===
                processing_start_time = time.time()

                # Concatenate all camera images horizontally
                if len(all_camera_images) == 1:
                    full_combined_image = all_camera_images[0]
                else:
                    full_combined_image = cv2.hconcat(all_camera_images)

                # Create frame metadata
                frame_metadata = self._create_frame_metadata(cameras_metadata)

                # Encode the combined image
                ret, buffer = cv2.imencode('.jpg', full_combined_image)
                if not ret:
                    print("[Image Server] Frame imencode failed.")
                    continue

                jpg_bytes = buffer.tobytes()
                frame_size = len(jpg_bytes)
                
                processing_end_time = time.time()
                
                # === MESSAGE CREATION PHASE ===
                message_creation_start = time.time()
                
                # Create message with metadata
                if self.Unit_Test:
                    timestamp = capture_start_time  # Use capture start time as frame timestamp
                    frame_id = self.frame_count
                    
                    # Add detailed timing information to metadata
                    frame_metadata["timing"] = {
                        "capture_start": capture_start_time,
                        "capture_duration": capture_end_time - capture_start_time,
                        "processing_duration": processing_end_time - processing_start_time,
                        "frame_size_bytes": frame_size
                    }
                    
                    # Serialize metadata to JSON
                    metadata_json = json.dumps(frame_metadata).encode('utf-8')
                    metadata_size = len(metadata_json)
                    
                    # Header: timestamp (8 bytes) + frame_id (4 bytes) + metadata_size (4 bytes)
                    header = struct.pack('dII', timestamp, frame_id, metadata_size)
                    message = header + metadata_json + jpg_bytes
                else:
                    # For non-test mode, still include metadata for stream parsing
                    metadata_json = json.dumps(frame_metadata).encode('utf-8')
                    metadata_size = len(metadata_json)
                    
                    # Header: metadata_size (4 bytes)
                    header = struct.pack('I', metadata_size)
                    message = header + metadata_json + jpg_bytes

                message_creation_end = time.time()
                
                # === TRANSMISSION PHASE ===
                transmission_start = time.time()
                self.socket.send(message)
                transmission_end = time.time()

                if self.Unit_Test:
                    # Calculate timing metrics
                    capture_duration = capture_end_time - capture_start_time
                    processing_duration = processing_end_time - processing_start_time
                    
                    self._update_performance_metrics(
                        capture_start_time, 
                        processing_duration, 
                        transmission_end, 
                        frame_size
                    )
                    self._print_performance_metrics(transmission_end)

        except KeyboardInterrupt:
            print("[Image Server] Interrupted by user.")
        finally:
            self._close()


if __name__ == "__main__":
    
    config = load_camera_config()    
    server = CameraServer(config, Unit_Test=False)
    server.send_process()
    