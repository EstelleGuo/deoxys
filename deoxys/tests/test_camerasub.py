import cv2
import time
from deoxys_vision.networking.camera_redis_interface import CameraRedisSubInterface
from deoxys.utils.cam_utils import load_camera_config

def main():
    camera_infos = load_camera_config()  # <- change path if needed
    subscribers = {}

    for cam_info in camera_infos:
        interface = CameraRedisSubInterface(
            camera_info=cam_info,
            redis_host="localhost",  # or "172.16.0.1" if needed
            use_color=True,
            use_depth=False,
        )
        interface.start()
        subscribers[cam_info.camera_name] = interface

    print("Subscribed to all cameras. Displaying images...")

    try:
        while True:
            for name, interface in subscribers.items():
                imgs = interface.get_img()
                if imgs["color"] is not None:
                    cv2.imshow(name, imgs["color"])
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        for interface in subscribers.values():
            interface.stop()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
