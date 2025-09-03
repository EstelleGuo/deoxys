import threading
import time
import numpy as np
from evdev import InputDevice, list_devices, ecodes
from deoxys.utils.transform_utils import rotation_matrix


def scale_to_control(x, min_raw=0, max_raw=255, min_v=-1.0, max_v=1.0, deadzone=1):
    """
    Normalize raw HID readings from [min_raw, max_raw] to [min_v, max_v], with deadzone.
    """
    center = (max_raw + min_raw) / 2
    if abs(x - center) <= deadzone:
        return 0.0
    x = (x - min_raw) / (max_raw - min_raw) * (max_v - min_v) + min_v
    return min(max(x, min_v), max_v)

class ZikwayGamepad:
    """
    Driver class for Zikway HID Gamepad.
    Mimics the interface of SpaceMouse driver.
    """

    AXIS_MAP = {
        ecodes.ABS_X: 'LY', #0 ~ 255  rotation around y 
        ecodes.ABS_Y: 'LX', #0 ~ 255  rotation around x
        ecodes.ABS_RX: 'RX', 
        ecodes.ABS_RY: 'RY',
        ecodes.ABS_HAT0X: 'DPad Y',  # left : -1, middle : 0, right : +1  translation along y
        ecodes.ABS_HAT0Y: 'DPad X',  # up : -1, middle : 0, down : +1  translation along x
        ecodes.ABS_Z: 'RZ', #0 ~ 255 rotation around z
        ecodes.ABS_RZ: 'Z', #0 ~ 255 translation along z
    }

    BUTTON_MAP = {
        ecodes.BTN_SOUTH: 'A',
        ecodes.BTN_EAST: 'B',
        ecodes.BTN_NORTH: 'X',
        ecodes.BTN_WEST: 'Y',
        ecodes.BTN_TL: 'LB',
        ecodes.BTN_TR: 'RB',
        ecodes.BTN_SELECT: 'Select',
        ecodes.BTN_START: 'Start',
        ecodes.BTN_THUMBL: 'LStick',
        ecodes.BTN_THUMBR: 'RStick',
    }

    def __init__(self, device_name='Zikway', pos_sensitivity=1.0, rot_sensitivity=1.0):
        print("Opening ZikwayGamepad device")
        self.device_path = None
        for dev_path in list_devices():
            dev = InputDevice(dev_path)
            if device_name in dev.name:
                self.device_path = dev_path
                break

        if self.device_path is None:
            raise RuntimeError(f"{device_name} not found")

        self.device = InputDevice(self.device_path)
        print(f"Connected to {self.device.name} at {self.device_path}")
        
        self.pos_sensitivity = pos_sensitivity
        self.rot_sensitivity = rot_sensitivity

        # 6-DOF variables
        self.x, self.y, self.z = 0, 0, 0
        self.roll, self.pitch, self.yaw = 0, 0, 0

        self._display_controls()
        self.single_click_and_hold = False

        self._control = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        # This state can be used to stop collection and reset robot to default joint position
        self._reset_state = 0
        self.rotation = np.array([[-1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, -1.0]])
        self._enabled = False
        self._running = True
        ###
        self._collecting = False

        self.buttons = {name: 0 for name in self.BUTTON_MAP.values()}

        self.thread = threading.Thread(target=self.run)
        self.thread.daemon = False
        self.thread.start()

    @staticmethod
    def _display_controls():
        """
        Method to pretty print controls for ZikwayGamepad.
        """
        def print_command(char, info):
            char += " " * (30 - len(char))
            print("{}\t{}".format(char, info))

        print("")
        print_command("Control", "Command")
        print_command("X button (press)", "close gripper")
        print_command("Y button (press)", "open gripper")
        print_command("Select button", "reset simulation")
        print_command("DPad (left/right)", "move arm in x direction")
        print_command("DPad (up/down)", "move arm in y direction")
        print_command("A button (press)", "start collecting")
        print_command("B button (press)", "stop collecting")
        print_command("ESC", "quit")
        print("")

    def _reset_internal_state(self):
        """
        Resets internal state of controller, except for the reset signal.
        """
        self.rotation = np.array([[-1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, -1.0]])
        self.x, self.y, self.z = 0, 0, 0
        self.roll, self.pitch, self.yaw = 0, 0, 0
        self._control = np.zeros(6)
        self.single_click_and_hold = False

    def start_control(self):
        """Enable reading inputs"""
        self._reset_internal_state()
        self._reset_state = 0
        self._enabled = True
        self._collecting = False

    def stop_control(self):
        """Disable reading inputs"""
        self._enabled = False
        self._running = False

    def close(self):
        """Stop and clean up"""
        self.stop_control()
        if hasattr(self, 'thread') and self.thread.is_alive():
            self.thread.join(timeout=1.0)

    def get_controller_state(self):
        """
        Grabs the current state of the gamepad.

        Returns:
            dict: A dictionary containing dpos, orn, unmodified orn, grasp, and reset
        """
        dpos = self.control[:3] * 0.005 * self.pos_sensitivity
        roll, pitch, yaw = self.control[3:] * 0.005 * self.rot_sensitivity

        drot1 = rotation_matrix(angle=-pitch, direction=[1.0, 0, 0], point=None)[:3, :3]
        drot2 = rotation_matrix(angle=roll, direction=[0, 1.0, 0], point=None)[:3, :3]
        drot3 = rotation_matrix(angle=yaw, direction=[0, 0, 1.0], point=None)[:3, :3]

        self.rotation = self.rotation.dot(drot1.dot(drot2.dot(drot3)))

        return dict(
            dpos=dpos,
            rotation=self.rotation,
            raw_drotation=np.array([roll, pitch, yaw]),
            grasp=self.control_gripper,
            reset=self._reset_state,
        )

    def run(self):
        """
        Listener method that keeps pulling new messages.
        """
        try:
            while self._running:
                try:
                    for event in self.device.read_loop():
                        if not self._enabled:
                            time.sleep(0.01)
                            continue
                        # Axis events
                        if event.type == ecodes.EV_ABS and event.code in self.AXIS_MAP:
                            name = self.AXIS_MAP[event.code]
                            value = event.value
                            if name == 'DPad X':
                                self.roll = 0
                            elif name == 'DPad Y':
                                self.pitch = 0
                            elif name == 'Z':
                                self.z = scale_to_control(value) * -1.0
                            elif name == 'LX':
                                self.x = scale_to_control(value) * -1.0
                            elif name == 'LY':
                                self.y = scale_to_control(value)
                            elif name == 'RZ':
                                self.yaw = scale_to_control(value)

                            self._control = [
                                self.x,
                                self.y,
                                self.z,
                                self.roll,
                                self.pitch,
                                self.yaw,
                            ]

                        # Button events
                        elif event.type == ecodes.EV_KEY and event.code in self.BUTTON_MAP:
                            btn_name = self.BUTTON_MAP[event.code]
                            self.buttons[btn_name] = event.value

                            # Gripper logic
                            if btn_name == 'X':
                                if event.value == 1:
                                    self.single_click_and_hold = True
                            elif btn_name == 'Y':
                                if event.value == 1:
                                    self.single_click_and_hold = False

                            # Reset logic
                            if btn_name == 'Select' and event.value == 1:
                                self._reset_state = 1
                                self._enabled = False
                                self._reset_internal_state()

                            if btn_name == 'A' and event.value == 1:
                                self._collecting = True

                            if btn_name == 'B' and event.value == 1:
                                self._collecting = False


                except Exception as e:
                    if self._running:
                        print(f"[ZikwayGamepad] Exception in run: {e}")
                    break
        except Exception as e:
            print(f"ZikwayGamepad thread error: {e}")
        finally:
            try:
                if hasattr(self, 'device'):
                    self.device.close()
            except Exception:
                pass

    @property
    def control(self):
        """
        Grabs current pose of Gamepad

        Returns:
            np.array: 6-DoF control value
        """
        return np.array(self._control)

    @property
    def control_gripper(self):
        """
        Maps internal states into gripper commands.

        Returns:
            float: Whether we're using single click and hold or not
        """
        if self.single_click_and_hold:
            return 1.0
        return 0

    def __del__(self):
        """Destructor to ensure proper cleanup."""
        self.close()
        print("ZikwayGamepad closed.")


if __name__ == "__main__":
    gamepad = ZikwayGamepad()
    gamepad.start_control()
    try:
        for i in range(1000):
            print(gamepad.control, gamepad.control_gripper)
            time.sleep(0.02)
    except KeyboardInterrupt:
        print("Interrupted by user")
    finally:
        gamepad.close()
