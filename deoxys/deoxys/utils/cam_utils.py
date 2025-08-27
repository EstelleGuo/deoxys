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
        return (f"CameraInfo(type={self.camera_type}, id={self.camera_id}, "
                f"name={self.camera_name}, serial={self.camera_serial_num}, "
                f"width={self.cfg.get('width', 'N/A')}, height={self.cfg.get('height', 'N/A')}, fps={self.cfg.get('fps', 'N/A')})")

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

        camera_config = {}
        if camera_type == "realsense":
            camera_config = {
            "width": config.get("cam_config", {}).get("realsense", {}).get("width", 640),
            "height": config.get("cam_config", {}).get("realsense", {}).get("height", 480),
            "fps": config.get("cam_config", {}).get("realsense", {}).get("fps", 30),
            "processing_preset": config.get("cam_config", {}).get("realsense", {}).get("processing_preset", 1),
            "depth": config.get("cam_config", {}).get("realsense", {}).get("depth", False)
            }

        elif camera_type == "opencv":
            camera_config = {
            "width": config.get("cam_config", {}).get("opencv", {}).get("width", 640),
            "height": config.get("cam_config", {}).get("opencv", {}).get("height", 480),
            "fps": config.get("cam_config", {}).get("opencv", {}).get("fps", 30)
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