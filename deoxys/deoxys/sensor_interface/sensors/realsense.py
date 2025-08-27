import numpy as np
from deoxys.sensor_interface.network import ZMQCameraPublisher
from deoxys.utils.cam_utils import FrequencyTimer, notify_component_start
import pyrealsense2 as rs
import time

class RealSenseCamera(object):
    def __init__(self, img_shape, fps, serial_number=None, enable_depth=False) -> None:
        """
        img_shape: [height, width]
        serial_number: serial number
        """
        self.img_shape = img_shape
        self.fps = fps
        self.serial_number = serial_number
        self.enable_depth = enable_depth

        align_to = rs.stream.color
        self.align = rs.align(align_to)
        self.init_realsense()

    def init_realsense(self):

        self.pipeline = rs.pipeline()
        config = rs.config()
        if self.serial_number is not None:
            config.enable_device(self.serial_number)

        config.enable_stream(rs.stream.color, self.img_shape[1], self.img_shape[0], rs.format.bgr8, self.fps)

        if self.enable_depth:
            config.enable_stream(rs.stream.depth, self.img_shape[1], self.img_shape[0], rs.format.z16, self.fps)

        profile = self.pipeline.start(config)
        self._device = profile.get_device()
        if self._device is None:
            print('[Image Server] pipe_profile.get_device() is None .')
        if self.enable_depth:
            assert self._device is not None
            depth_sensor = self._device.first_depth_sensor()
            self.g_depth_scale = depth_sensor.get_depth_scale()

        self.intrinsics = profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()

    def get_frame(self):
        frames = self.pipeline.wait_for_frames()
        aligned_frames = self.align.process(frames)
        color_frame = aligned_frames.get_color_frame()

        if self.enable_depth:
            depth_frame = aligned_frames.get_depth_frame()

        if not color_frame:
            return None

        color_image = np.asanyarray(color_frame.get_data())
        # color_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)
        depth_image = np.asanyarray(depth_frame.get_data()) if self.enable_depth else None
        return color_image, depth_image

    def release(self):
        self.pipeline.stop()

