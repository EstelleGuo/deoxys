import cv2
import zmq
import numpy as np
import time
import struct
from collections import deque
from multiprocessing import shared_memory
import threading
from deoxys.utils.cam_utils import load_camera_config, CameraInfo

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
        config example1:
        {
            cam_host : 192.168.1.113
            cam_port : 10001

            cam_infos:
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

        print(config)

        self.running = True
        self._image_show = image_show
        self.cam_configs = config
        self.camera_infos = config.get('camera_infos', [])
        self._port = config.get('camera_port', 10001)
        self._server_address = config.get('camera_host', 'localhost')
        self.Unit_Test = Unit_Test

        ## Drop out the Shared Memory should work
        self.img_contents = {}
        for cam_info in self.camera_infos:
            img_shape = getattr(cam_info, 'image_shape', [getattr(cam_info.cfg, 'height', 480), getattr(cam_info.cfg, 'width', 640), 3])
            cam_name = getattr(cam_info, 'camera_name', 'camera_0')
            cam_id = getattr(cam_info, 'camera_id', 0)
            image_shm = shared_memory.SharedMemory(create=True, size=np.prod(img_shape) * np.uint8().itemsize)
            img_array = np.ndarray(img_shape, dtype=np.uint8, buffer=image_shm.buf)
            serial_number = getattr(cam_info, 'camera_serial_num', None)
            self.img_contents[cam_name] = {
                'image_shape': img_shape,
                'cam_name': cam_name,
                'cam_id': cam_id,
                'cam_serial_num': serial_number,
                'img_array': img_array,
                'image_shm': image_shm
            }

        # Performance evaluation parameters
        self._enable_performance_eval = Unit_Test
        if self._enable_performance_eval:
            self._init_performance_metrics()

    def _init_performance_metrics(self):
        self._frame_count = 0  # Total frames received
        self._last_frame_id = -1  # Last received frame ID

        # Real-time FPS calculation using a time window
        self._time_window = 1.0  # Time window size (in seconds)
        self._frame_times = deque()  # Timestamps of frames received within the time window

        # Data transmission quality metrics
        self._latencies = deque()  # Latencies of frames within the time window
        self._lost_frames = 0  # Total lost frames
        self._total_frames = 0  # Expected total frames based on frame IDs

    def _update_performance_metrics(self, timestamp, frame_id, receive_time):
        # Update latency
        latency = receive_time - timestamp

        
        self._latencies.append(latency)

        # Remove latencies outside the time window
        while self._latencies and self._frame_times and self._latencies[0] < receive_time - self._time_window:
            self._latencies.popleft()

        # Update frame times
        self._frame_times.append(receive_time)
        # Remove timestamps outside the time window
        while self._frame_times and self._frame_times[0] < receive_time - self._time_window:
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

            # Calculate latency metrics
            if self._latencies:
                avg_latency = sum(self._latencies) / len(self._latencies)
                max_latency = max(self._latencies)
                min_latency = min(self._latencies)
                jitter = max_latency - min_latency
            else:
                avg_latency = max_latency = min_latency = jitter = 0

            # Calculate lost frame rate
            lost_frame_rate = (self._lost_frames / self._total_frames) * 100 if self._total_frames > 0 else 0

            print(f"[Image Client] Real-time FPS: {real_time_fps:.2f}, Avg Latency: {avg_latency*1000:.2f} ms, Max Latency: {max_latency*1000:.2f} ms, \
                  Min Latency: {min_latency*1000:.2f} ms, Jitter: {jitter*1000:.2f} ms, Lost Frame Rate: {lost_frame_rate:.2f}%")
    
    def _close(self):
        # Release all shared memory objects
        for cam_name, img_info in getattr(self, 'img_contents', {}).items():
            shm = img_info.get('image_shm', None)
            if shm is not None:
                try:
                    shm.close()
                    shm.unlink()
                    print(f"Released shared memory for {cam_name}")
                except Exception as e:
                    print(f"Error releasing shared memory for {cam_name}: {e}")
                    
        self._socket.close()
        self._context.term()
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
                # Receive message
                message = self._socket.recv()
                receive_time = time.time()
                
                if self._enable_performance_eval:
                    header_size = struct.calcsize('dI')
                    try:
                        # Attempt to extract header and image data
                        header = message[:header_size]
                        jpg_bytes = message[header_size:]
                        timestamp, frame_id = struct.unpack('dI', header)
                    except struct.error as e:
                        print(f"[Image Client] Error unpacking header: {e}, discarding message.")
                        continue
                else:
                    # No header, entire message is image data
                    jpg_bytes = message
                # Decode image
                np_img = np.frombuffer(jpg_bytes, dtype=np.uint8)
                current_image = cv2.imdecode(np_img, cv2.IMREAD_COLOR)
                if current_image is None:
                    print("[Image Client] Failed to decode image.")
                    continue

                for image_content in self.img_contents.values():
                    cam_width = image_content['image_shape'][1]
                    cam_id = image_content['cam_id']
                    np.copyto(image_content['img_array'], np.array(current_image[:, cam_id * cam_width : (cam_id + 1) * cam_width]))

                if self._image_show:
                    # Display each camera image in a separate window
                    for cam_name, image_content in self.img_contents.items():
                        img = image_content['img_array']
                        serial_num = image_content.get('cam_serial_num', 'unknown')
                        window_name = f"Camera: {cam_name} (SN: {serial_num})"
                        cv2.imshow(window_name, img)

                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        self.running = False

                if self._enable_performance_eval:
                    self._update_performance_metrics(timestamp, frame_id, receive_time)
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


    while True:
        time.sleep(1)
        print("Image client is running... Press Ctrl+C to stop.")

