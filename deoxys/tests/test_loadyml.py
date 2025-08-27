import yaml
import os
import yaml
from deoxys import config_root
import cv2
import time
from deoxys.utils.cam_utils import load_camera_config, CameraInfo

def main():
    config = load_camera_config()
    print(f"Camera Host: {config['camera_host']}")
    print(f"Camera Port: {config['camera_port']}")
    print("Camera Infos:")
    for cam in config['camera_infos']:
        print(cam)

if __name__ == '__main__':
	main()
