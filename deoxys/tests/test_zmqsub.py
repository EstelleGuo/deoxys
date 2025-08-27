import cv2
import time

from deoxys.utils.cam_utils import load_camera_config
from deoxys.sensor_interface.network import ZMQCameraSubInterface

def main():
    camera_infos = load_camera_config()  # <- change path if needed
    subscribers = {}
    i = 0
    for cam_info in camera_infos:
        interface = ZMQCameraSubscriber(
            host="192.168.1.113",
            port=10006 + i,      # or 10007 for the second set
            topic_type="RGB"
        )
        interface.start()
        subscribers[cam_info.camera_name] = interface
        i += 1


    print("Subscribed to all cameras (ZMQ). Displaying images... Press 'q' to quit.")

    try:
        while True:
            for name, interface in subscribers.items():
                img, timestamp = interface.recv_rgb_image()
                if img is not None:
                    cv2.imshow(name, img)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            time.sleep(0.005)
    finally:
        for interface in subscribers.values():
            interface.stop()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
