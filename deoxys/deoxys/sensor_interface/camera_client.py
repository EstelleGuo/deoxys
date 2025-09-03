import cv2
import zmq
import numpy as np
import time
import struct
import json
from collections import deque
from multiprocessing import shared_memory
import threading
import os
from deoxys.sensor_interface.sensors.realsense_v1 import RealSenseCamera_V1
from deoxys.utils.cam_utils import load_camera_config
from deoxys.sensor_interface.sensors.opencv_cam import OpenCVCamera_V1

class CameraClient:
    def __init__(self, config , image_show = False, Unit_Test = False):
        """        
        image_show: Whether to display received images in real time.

        server_address: The ip address to execute the image server script.

        port: The port number to bind to. It should be the same as the image server.

        Unit_Test: When both server and client are True, it can be used to test the image transfer latency, \
                   network jitter, frame loss rate and other information.
        """
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

        print(config)

        self.running = True
        self._image_show = image_show
        self.cam_configs = config
        self.camera_infos = config.get('camera_infos', [])
        self._port = config.get('camera_port', 10001)
        self._server_address = config.get('camera_host', 'localhost')
        self.Unit_Test = Unit_Test

        # Sort cameras by cam_id to ensure proper order for image splitting
        self.camera_infos = sorted(self.camera_infos, key=lambda cam: getattr(cam, 'camera_id', 0))

        # Create image containers for each camera and stream type
        self.img_contents = {}
        for cam_info in self.camera_infos:
            # Get image shape from camera config
            cam_config = getattr(cam_info, 'cfg', {})
            height = cam_config.get('height', 480)
            width = cam_config.get('width', 640)
            
            cam_name = getattr(cam_info, 'camera_name', 'camera_0')
            cam_id = getattr(cam_info, 'camera_id', 0)
            cam_type = getattr(cam_info, 'camera_type', 'opencv')
            serial_number = getattr(cam_info, 'camera_serial_num', None)
            
            # Create shared memory for different stream types
            self.img_contents[cam_name] = {
                'cam_name': cam_name,
                'cam_id': cam_id,
                'cam_type': cam_type,
                'cam_serial_num': serial_number,
                'streams': {}
            }
            
            # Create shared memory for each possible stream type
            stream_types = ['color']
            if cam_type == 'realsense':
                # Check what streams are enabled in config
                color_sensor = cam_config.get('color_sensor', {})
                depth_sensor = cam_config.get('depth_sensor', {})
                ir_sensors = cam_config.get('ir_sensors', {})
                
                if color_sensor.get('enable', True):
                    stream_types.append('color')
                if depth_sensor.get('enable', False):
                    stream_types.append('depth')
                if ir_sensors.get('enable', False):
                    stream_types.extend(['ir_left', 'ir_right'])
            
            # Remove duplicates and create shared memory for each stream
            stream_types = list(set(stream_types))
            
            for stream_type in stream_types:
                img_shape = [height, width, 3]  # All streams converted to 3-channel
                
                shm_name = f"{cam_name}_{stream_type}"
                try:
                    image_shm = shared_memory.SharedMemory(name=shm_name, create=True, size=np.prod(img_shape) * np.uint8().itemsize)
                except FileExistsError:
                    # If it already exists, connect to it
                    image_shm = shared_memory.SharedMemory(name=shm_name)
                
                img_array = np.ndarray(img_shape, dtype=np.uint8, buffer=image_shm.buf)
                
                self.img_contents[cam_name]['streams'][stream_type] = {
                    'image_shape': img_shape,
                    'img_array': img_array,
                    'image_shm': image_shm
                }

        # Performance evaluation parameters
        self._enable_performance_eval = Unit_Test
        if self._enable_performance_eval:
            self._init_performance_metrics()

    def _parse_multi_stream_image(self, combined_image, frame_metadata):
        """
        Parse the combined image back into individual camera streams
        
        Args:
            combined_image: The concatenated image from server
            frame_metadata: Metadata describing the image structure
        
        Returns:
            dict: {cam_name: {stream_type: image_array}}
        """
        parsed_images = {}
        
        for cam_data in frame_metadata["cameras"]:
            cam_name = cam_data["cam_name"]
            start_x = cam_data["start_x"]
            width = cam_data["width"]
            height = cam_data["height"]
            streams = cam_data["streams"]
            
            # Extract this camera's portion from the combined image
            camera_section = combined_image[:height, start_x:start_x+width]
            
            # Split the camera section into individual streams
            parsed_images[cam_name] = {}
            
            if len(streams) == 1:
                # Single stream for this camera
                stream_type = streams[0]
                parsed_images[cam_name][stream_type] = camera_section
            else:
                # Multiple streams - split horizontally
                stream_width = width // len(streams)
                for i, stream_type in enumerate(streams):
                    stream_start_x = i * stream_width
                    stream_end_x = (i + 1) * stream_width
                    stream_image = camera_section[:, stream_start_x:stream_end_x]
                    parsed_images[cam_name][stream_type] = stream_image
        
        return parsed_images

    def _init_performance_metrics(self):
        self._frame_count = 0  # Total frames received
        self._last_frame_id = -1  # Last received frame ID

        # Real-time FPS calculation using a time window
        self._time_window = 1.0  # Time window size (in seconds)
        self._frame_times = deque()  # Timestamps of frames received within the time window

        # Enhanced performance tracking
        self._receive_times = deque()  # Frame receive timestamps
        self._processing_times = deque()  # Client processing durations
        self._display_times = deque()  # Frame display timestamps
        self._decode_times = deque()  # Image decode durations
        self._parse_times = deque()  # Image parsing durations

        # Data transmission quality metrics
        self._latencies = deque()  # End-to-end latencies (capture to receive)
        self._network_latencies = deque()  # Network transmission latencies
        self._lost_frames = 0  # Total lost frames
        self._total_frames = 0  # Expected total frames based on frame IDs
        self._frame_sizes = deque()  # Received frame sizes

    def _update_performance_metrics(self, capture_timestamp, frame_id, receive_time, decode_duration, parse_duration, display_time, frame_size, server_timing=None):
        # Update end-to-end latency (capture to receive)
        end_to_end_latency = receive_time - capture_timestamp
        self._latencies.append(end_to_end_latency)

        # Update timing deques
        self._receive_times.append(receive_time)
        self._decode_times.append(decode_duration)
        self._parse_times.append(parse_duration)
        self._display_times.append(display_time)
        self._frame_sizes.append(frame_size)
        
        # Calculate network latency from server timing if available
        if server_timing and "capture_start" in server_timing:
            server_transmission_time = capture_timestamp  # Server puts capture_start as timestamp
            network_latency = receive_time - server_transmission_time
            self._network_latencies.append(network_latency)

        # Remove metrics outside the time window
        current_time = receive_time
        while self._latencies and self._receive_times and self._latencies[0] < current_time - self._time_window:
            self._latencies.popleft()
        while self._receive_times and self._receive_times[0] < current_time - self._time_window:
            self._receive_times.popleft()
        while self._decode_times and len(self._decode_times) > len(self._receive_times):
            self._decode_times.popleft()
        while self._parse_times and len(self._parse_times) > len(self._receive_times):
            self._parse_times.popleft()
        while self._display_times and len(self._display_times) > len(self._receive_times):
            self._display_times.popleft()
        while self._frame_sizes and len(self._frame_sizes) > len(self._receive_times):
            self._frame_sizes.popleft()
        while self._network_latencies and len(self._network_latencies) > len(self._receive_times):
            self._network_latencies.popleft()

        # Update frame times
        self._frame_times.append(receive_time)
        # Remove timestamps outside the time window
        while self._frame_times and self._frame_times[0] < current_time - self._time_window:
            self._frame_times.popleft()

        # Update frame counts for lost frame calculation
        expected_frame_id = self._last_frame_id + 1 if self._last_frame_id != -1 else frame_id
        if frame_id != expected_frame_id:
            lost = frame_id - expected_frame_id
            if lost < 0:
                print(f"[Image Client] Received out-of-order frame ID: {frame_id}")
            else:
                self._lost_frames += lost
                print(f"[Image Client] Detected lost frames: {lost}, Expected frame ID: {expected_frame_id}, Received frame ID: {frame_id}")
        self._last_frame_id = frame_id
        self._total_frames = frame_id + 1

        self._frame_count += 1

    def _print_performance_metrics(self, receive_time):
        if self._frame_count % 30 == 0:
            # Calculate real-time FPS
            real_time_fps = len(self._frame_times) / self._time_window if self._time_window > 0 else 0

            # Calculate end-to-end latency metrics
            if self._latencies:
                avg_latency = sum(self._latencies) / len(self._latencies)
                max_latency = max(self._latencies)
                min_latency = min(self._latencies)
                jitter = max_latency - min_latency
            else:
                avg_latency = max_latency = min_latency = jitter = 0
            
            # Calculate network latency metrics
            if self._network_latencies:
                avg_network_latency = sum(self._network_latencies) / len(self._network_latencies)
                max_network_latency = max(self._network_latencies)
                min_network_latency = min(self._network_latencies)
            else:
                avg_network_latency = max_network_latency = min_network_latency = 0

            # Calculate processing metrics
            if self._decode_times:
                recent_decode_times = list(self._decode_times)[-min(30, len(self._decode_times)):]
                avg_decode_time = sum(recent_decode_times) / len(recent_decode_times)
                max_decode_time = max(recent_decode_times)
                min_decode_time = min(recent_decode_times)
            else:
                avg_decode_time = max_decode_time = min_decode_time = 0
                
            if self._parse_times:
                recent_parse_times = list(self._parse_times)[-min(30, len(self._parse_times)):]
                avg_parse_time = sum(recent_parse_times) / len(recent_parse_times)
                max_parse_time = max(recent_parse_times)
                min_parse_time = min(recent_parse_times)
            else:
                avg_parse_time = max_parse_time = min_parse_time = 0

            # Calculate frame size metrics
            if self._frame_sizes:
                recent_frame_sizes = list(self._frame_sizes)[-min(30, len(self._frame_sizes)):]
                avg_frame_size = sum(recent_frame_sizes) / len(recent_frame_sizes)
                max_frame_size = max(recent_frame_sizes)
                min_frame_size = min(recent_frame_sizes)
                total_data_received = sum(self._frame_sizes) / (1024 * 1024)  # MB
            else:
                avg_frame_size = max_frame_size = min_frame_size = total_data_received = 0

            # Calculate lost frame rate
            lost_frame_rate = (self._lost_frames / self._total_frames) * 100 if self._total_frames > 0 else 0

            print(f"[Image Client] === Performance Metrics ===")
            print(f"  FPS: {real_time_fps:.2f} | Frames: {self._frame_count} | Lost: {lost_frame_rate:.2f}%")
            print(f"  End-to-End Latency: Avg={avg_latency*1000:.1f}ms, Max={max_latency*1000:.1f}ms, Min={min_latency*1000:.1f}ms")
            print(f"  Network Latency: Avg={avg_network_latency*1000:.1f}ms, Max={max_network_latency*1000:.1f}ms, Min={min_network_latency*1000:.1f}ms")
            print(f"  Decode Time: Avg={avg_decode_time*1000:.1f}ms, Max={max_decode_time*1000:.1f}ms, Min={min_decode_time*1000:.1f}ms")
            print(f"  Parse Time: Avg={avg_parse_time*1000:.1f}ms, Max={max_parse_time*1000:.1f}ms, Min={min_parse_time*1000:.1f}ms")
            print(f"  Jitter: {jitter*1000:.1f}ms")
            print(f"  Frame Size: Avg={avg_frame_size/1024:.1f}KB, Max={max_frame_size/1024:.1f}KB, Min={min_frame_size/1024:.1f}KB")
            print(f"  Total Data Received: {total_data_received:.2f}MB")
            print(f"  =====================================")
            
            # Calculate receive bandwidth
            if len(self._frame_times) > 1:
                time_span = self._frame_times[-1] - self._frame_times[0]
                if time_span > 0:
                    bandwidth_mbps = (sum(recent_frame_sizes) * 8) / (time_span * 1024 * 1024)  # Mbps
                    print(f"  Receive Bandwidth: {bandwidth_mbps:.2f} Mbps")
    
    def stop(self):
        """Gracefully stop the camera client"""
        print("[Image Client] Stopping camera client...")
        self.running = False
        
    def _close(self):
        if hasattr(self, '_socket'):
            self._socket.close()
        if hasattr(self, '_context'):
            self._context.term()
        
        # Clean up shared memory
        for cam_name, cam_content in self.img_contents.items():
            for stream_type, stream_content in cam_content['streams'].items():
                try:
                    stream_content['image_shm'].close()
                    stream_content['image_shm'].unlink()  # Remove the shared memory
                except Exception as e:
                    print(f"[Image Client] Error cleaning up shared memory for {cam_name}_{stream_type}: {e}")
        
        if self._image_show:
            cv2.destroyAllWindows()
        print("Image client has been closed.")

    def receive_process(self):
        # Set up ZeroMQ context and socket
        self._context = zmq.Context()
        self._socket = self._context.socket(zmq.SUB)
        self._socket.connect(f"tcp://{self._server_address}:{self._port}")
        self._socket.setsockopt_string(zmq.SUBSCRIBE, "")

        print("\nImage client has started, waiting to receive data...")
        try:
            while self.running:
                # === MESSAGE RECEIVE PHASE ===
                receive_start = time.time()
                message = self._socket.recv()
                receive_time = time.time()
                
                frame_size = len(message)
                
                # === MESSAGE PARSING PHASE ===
                parse_start = time.time()
                
                # Parse message header
                server_timing = None
                if self._enable_performance_eval:
                    # Unit test mode: timestamp + frame_id + metadata_size + metadata + image
                    header_size = struct.calcsize('dII')
                    try:
                        header = message[:header_size]
                        timestamp, frame_id, metadata_size = struct.unpack('dII', header)
                        
                        metadata_start = header_size
                        metadata_end = metadata_start + metadata_size
                        metadata_json = message[metadata_start:metadata_end]
                        jpg_bytes = message[metadata_end:]
                        
                    except (struct.error, json.JSONDecodeError) as e:
                        print(f"[Image Client] Error parsing unit test message: {e}")
                        continue
                else:
                    # Normal mode: metadata_size + metadata + image
                    try:
                        metadata_size = struct.unpack('I', message[:4])[0]
                        metadata_start = 4
                        metadata_end = metadata_start + metadata_size
                        metadata_json = message[metadata_start:metadata_end]
                        jpg_bytes = message[metadata_end:]
                        timestamp = None
                        frame_id = 0
                    except (struct.error, json.JSONDecodeError) as e:
                        print(f"[Image Client] Error parsing message: {e}")
                        continue
                
                # Parse metadata
                try:
                    frame_metadata = json.loads(metadata_json.decode('utf-8'))
                    server_timing = frame_metadata.get("timing", {})
                except json.JSONDecodeError as e:
                    print(f"[Image Client] Error decoding metadata JSON: {e}")
                    continue
                
                parse_end = time.time()
                parse_duration = parse_end - parse_start
                
                # === IMAGE DECODE PHASE ===
                decode_start = time.time()
                np_img = np.frombuffer(jpg_bytes, dtype=np.uint8)
                combined_image = cv2.imdecode(np_img, cv2.IMREAD_COLOR)
                if combined_image is None:
                    print("[Image Client] Failed to decode image.")
                    continue
                decode_end = time.time()
                decode_duration = decode_end - decode_start

                # === IMAGE PARSING PHASE ===
                stream_parse_start = time.time()
                
                # Parse the combined image into individual camera streams
                parsed_images = self._parse_multi_stream_image(combined_image, frame_metadata)
                
                # Copy parsed images to shared memory
                for cam_name, stream_images in parsed_images.items():
                    if cam_name in self.img_contents:
                        for stream_type, stream_image in stream_images.items():
                            if stream_type in self.img_contents[cam_name]['streams']:
                                # Copy to shared memory
                                np.copyto(self.img_contents[cam_name]['streams'][stream_type]['img_array'], stream_image)

                stream_parse_end = time.time()
                stream_parse_duration = stream_parse_end - stream_parse_start

                # === DISPLAY PHASE ===
                display_start = time.time()
                
                if self._image_show:
                    # Display each camera's streams in separate windows
                    for cam_name, cam_content in self.img_contents.items():
                        serial_num = cam_content.get('cam_serial_num', 'unknown')
                        
                        for stream_type, stream_content in cam_content['streams'].items():
                            img = stream_content['img_array']
                            window_name = f"{cam_name}_{stream_type} (SN: {serial_num})"
                            cv2.imshow(window_name, img)

                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        self.running = False

                display_end = time.time()

                # === PERFORMANCE METRICS UPDATE ===
                if self._enable_performance_eval and timestamp is not None:
                    total_parse_duration = parse_duration + stream_parse_duration
                    
                    self._update_performance_metrics(
                        timestamp, 
                        frame_id, 
                        receive_time, 
                        decode_duration,
                        total_parse_duration,
                        display_end,
                        frame_size,
                        server_timing
                    )
                    self._print_performance_metrics(receive_time)

        except KeyboardInterrupt:
            print("Image client interrupted by user.")
        except Exception as e:
            print(f"[Image Client] An error occurred while receiving data: {e}")
        finally:
            self._close()

if __name__ == "__main__":
    # example
    # Initialize the client with performance evaluation enabled
    config = load_camera_config()   
    client = CameraClient(config, image_show = True, Unit_Test=False) # local test
    
    image_receive_thread = threading.Thread(target = client.receive_process, daemon = True)
    image_receive_thread.daemon = True
    image_receive_thread.start()

    try:
        while True:
            time.sleep(1)
            print("Image client is running... Press Ctrl+C to stop.")
    except KeyboardInterrupt:
        print("\nReceived interrupt signal. Shutting down gracefully...")
        client.stop()
        # Give the thread a moment to finish
        image_receive_thread.join(timeout=2.0)
        print("Camera client stopped.")

