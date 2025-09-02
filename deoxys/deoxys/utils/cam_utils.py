import os
import yaml
from deoxys import config_root
import cv2
import time

class CameraInfo:
    def __init__(self, camera_type, 
                       camera_name, 
                       camera_config, 
                       camera_id=None, 
                       camera_serial_num=None):
        self.camera_type = camera_type
        self.camera_id = camera_id
        self.camera_name = camera_name
        self.camera_serial_num = camera_serial_num
        self.cfg = camera_config

    def __repr__(self):
        # Build a more detailed representation
        base_info = (f"CameraInfo(type={self.camera_type}, id={self.camera_id}, "
                    f"name={self.camera_name}, serial={self.camera_serial_num})")
        
        # Add configuration details
        config_details = []
        config_details.append(f"  Resolution: {self.cfg.get('width', 'N/A')}x{self.cfg.get('height', 'N/A')} @ {self.cfg.get('fps', 'N/A')}fps")
        
        # Stream enable status
        streams = []
        if self.cfg.get('enable_color', True):
            streams.append("color")
        if self.cfg.get('enable_depth', False):
            streams.append("depth") 
        if self.cfg.get('enable_ir', False):
            streams.append("ir")
        config_details.append(f"  Enabled streams: {', '.join(streams) if streams else 'none'}")
        
        # Color sensor config (use correct key)
        color_sensor_config = self.cfg.get('color_sensor', {})
        if color_sensor_config:
            color_settings = []
            for key, value in color_sensor_config.items():
                color_settings.append(f"{key}={value}")
            config_details.append(f"  Color sensor: {{{', '.join(color_settings)}}}")
        else:
            config_details.append("  Color sensor: {}")
        
        # Depth sensor config (use correct key)
        depth_sensor_config = self.cfg.get('depth_sensor', {})
        if depth_sensor_config:
            depth_settings = []
            for key, value in depth_sensor_config.items():
                depth_settings.append(f"{key}={value}")
            config_details.append(f"  Depth sensor: {{{', '.join(depth_settings)}}}")
        else:
            config_details.append("  Depth sensor: {}")
        
        # IR sensor config (use correct key)
        ir_sensors_config = self.cfg.get('ir_sensors', {})
        if ir_sensors_config:
            ir_settings = []
            for key, value in ir_sensors_config.items():
                ir_settings.append(f"{key}={value}")
            config_details.append(f"  IR sensors: {{{', '.join(ir_settings)}}}")
        else:
            config_details.append("  IR sensors: {}")
        
        # Combine base info with details
        return base_info + "\n" + "\n".join(config_details)


def load_camera_config(yaml_path=None):
    if yaml_path is None:
       yaml_path = os.path.join(config_root, "camera_setup_config.yml")
    with open(yaml_path, "r") as f:
        config = yaml.safe_load(f)
    
    camera_infos = []
    camera_host = config.get("cam_host", "localhost")
    camera_port = int(config.get("cam_port", 10001))

    for entry in config.get("cam_infos", []):
        camera_type = entry.get("type", "unknown")
        camera_id = entry.get("cam_id", 0)
        camera_name = entry.get("name", "unknown")
        camera_serial_num = entry.get("cam_serial_num", "unknown")
        cam_config = entry.get("cam_config", {})

        camera_config = {}
        if camera_type == "realsense":
            # Parse hierarchical RealSense config
            color_sensor = cam_config.get("color_sensor", {})
            depth_sensor = cam_config.get("depth_sensor", {})
            ir_sensors = cam_config.get("ir_sensors", {})
            
            camera_config = {
                # Basic settings
                "width": cam_config.get("width", 640),
                "height": cam_config.get("height", 480),
                "fps": cam_config.get("fps", 30),
                "processing_preset": cam_config.get("processing_preset", 1),
                
                # Hierarchical sensor configuration (NEW)
                "color_sensor": color_sensor,
                "depth_sensor": depth_sensor,
                "ir_sensors": ir_sensors,
                
                # Flat configuration for backward compatibility
                "enable_color": color_sensor.get("enable", True),
                "enable_depth": depth_sensor.get("enable", False),
                "enable_ir": ir_sensors.get("enable", False),
                
                # Color sensor configuration (flattened)
                "color_auto_exposure": color_sensor.get("auto_exposure", True),
                "color_auto_white_balance": color_sensor.get("auto_white_balance", True),
                "color_exposure": color_sensor.get("exposure", 166),
                "color_gain": color_sensor.get("gain", 16),
                "color_white_balance": color_sensor.get("white_balance", 4000),
                "color_brightness": color_sensor.get("brightness", 32),
                "color_contrast": color_sensor.get("contrast", 32),
                "color_saturation": color_sensor.get("saturation", 32),
                
                # IR sensor configuration (flattened)
                "ir_auto_exposure": ir_sensors.get("auto_exposure", True),
                "ir_exposure": ir_sensors.get("exposure", 8500),
                "ir_gain": ir_sensors.get("gain", 16),
                "ir_disable_emitter": ir_sensors.get("disable_emitter", False),
                "ir_laser_power": ir_sensors.get("laser_power", 360),
            }

        elif camera_type == "opencv":
            camera_config = {
                "width": cam_config.get("opencv", {}).get("width", 640),
                "height": cam_config.get("opencv", {}).get("height", 480),
                "fps": cam_config.get("opencv", {}).get("fps", 30)
            }

        cam = CameraInfo(
            camera_type,
            camera_name,
            camera_config,
            camera_id,
            camera_serial_num
        )

        camera_infos.append(cam)

    return {'camera_host': camera_host, 'camera_port': camera_port, 'camera_infos': camera_infos}


def resize_img(img, camera_type, img_w=128, img_h=128, offset_w=0, offset_h=0):

    if camera_type == "k4a":
        resized_img = cv2.resize(img, (0, 0), fx=0.2, fy=0.2)
        w = resized_img.shape[0]
        h = resized_img.shape[1]

    if camera_type == "rs":
        resized_img = cv2.resize(img, (0, 0), fx=0.2, fy=0.3)
        w = resized_img.shape[0]
        h = resized_img.shape[1]

    resized_img = resized_img[
        w // 2 - img_w // 2 : w // 2 + img_w // 2,
        h // 2 - img_h // 2 : h // 2 + img_h // 2,
        :,
    ]
    return resized_img


def notify_component_start(component_name):
    print("***************************************************************")
    print("     Starting {} component".format(component_name))
    print("***************************************************************")


class FrequencyTimer(object):
    def __init__(self, frequency_rate):
        self.time_available = 1e9 / frequency_rate

    def start_loop(self):
        self.start_time = time.time_ns()

    def check_time(self, frequency_rate):
        # if prev_check_time variable doesn't exist, create it
        if not hasattr(self, "prev_check_time"):
            self.prev_check_time = self.start_time

        curr_time = time.time_ns()
        if (curr_time - self.prev_check_time) > 1e9 / frequency_rate:
            self.prev_check_time = curr_time
            return True
        return False

    def end_loop(self):
        wait_time = self.time_available + self.start_time

        while time.time_ns() < wait_time:
            continue