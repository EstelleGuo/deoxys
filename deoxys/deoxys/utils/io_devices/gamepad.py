import threading
import time
import numpy as np
from evdev import InputDevice, list_devices, ecodes
from deoxys.utils.transform_utils import rotation_matrix


def scale_to_control(x, min_raw=0, max_raw=255, min_v=-1.0, max_v=1.0, deadzone=5):
    center = (max_raw + min_raw) / 2
    if abs(x - center) <= deadzone:
        return 0.0
    x = (x - min_raw) / (max_raw - min_raw) * (max_v - min_v) + min_v
    return min(max(x, min_v), max_v)

def scale_centered(x, amin, amax, deadzone=8):
    """RX/RY 这类有中心点的轴 → [-1, 1]"""
    center = (amax + amin) / 2.0
    if abs(x - center) <= deadzone:
        return 0.0
    return float(np.clip((x - center) / ((amax - amin) / 2.0), -1.0, 1.0))

def scale_trigger(x, amin, amax, deadzone=8):
    """L2/R2 这类单向轴 → [0, 1]"""
    if x - amin <= deadzone:
        return 0.0
    return float(np.clip((x - amin) / max(1.0, (amax - amin)), 0.0, 1.0))

class ZikwayGamepad:  # actually PlayStationGamepad, but keep old name for compatibility
    """
    Driver class for Sony PlayStation controllers (DualShock 4 / DualSense).
    Mimics the interface of SpaceMouse driver and your Zikway driver.
    """

    # Typical Linux evdev codes for DS4/DualSense:
    # Axes:
    #   ABS_X / ABS_Y        -> Left stick (X/Y)
    #   ABS_RX / ABS_RY      -> Right stick (X/Y)
    #   ABS_Z / ABS_RZ       -> L2 / R2 analog (0..255)
    #   ABS_HAT0X / ABS_HAT0Y-> D-Pad (−1,0,+1)
    # Buttons:
    #   BTN_SOUTH/EAST/NORTH/WEST -> Cross/Circle/Triangle/Square
    #   BTN_TL / BTN_TR           -> L1/R1
    #   BTN_THUMBL / BTN_THUMBR   -> L3/R3
    #   BTN_SELECT / BTN_START    -> Share/Options (names vary by kernel)
    #   BTN_MODE                  -> PS button
    #   BTN_TL2 / BTN_TR2         -> L2/R2 digital click (optional; analog is on axes)

    AXIS_MAP = {
        ecodes.ABS_X: 'LX',          # -1..+1 left/right
        ecodes.ABS_Y: 'LY',          # -1..+1 up/down (note: usually up is negative)
        ecodes.ABS_RX: 'RX',
        ecodes.ABS_RY: 'RY',
        ecodes.ABS_HAT0X: 'DPad X',  # -1 left, 0 center, +1 right
        ecodes.ABS_HAT0Y: 'DPad Y',  # -1 up,   0 center, +1 down
        ecodes.ABS_Z: 'L2',          # 0..255
        ecodes.ABS_RZ: 'R2',         # 0..255
    }

    BUTTON_MAP = {
        ecodes.BTN_SOUTH: 'Cross',      # A on Xbox; use for "close gripper"
        ecodes.BTN_EAST: 'Circle',      # B on Xbox; use for "stop collecting"
        ecodes.BTN_NORTH: 'Triangle',   # Y on Xbox; use for "open gripper"
        ecodes.BTN_WEST: 'Square',      # X on Xbox
        ecodes.BTN_TL: 'L1',
        ecodes.BTN_TR: 'R1',
        ecodes.BTN_SELECT: 'Share',     # may appear as BTN_SELECT
        ecodes.BTN_START: 'Options',    # may appear as BTN_START
        ecodes.BTN_THUMBL: 'L3',
        ecodes.BTN_THUMBR: 'R3',
        # Optional extras if your kernel exposes them:
        # ecodes.BTN_MODE: 'PS',
        # ecodes.BTN_TL2: 'L2_click',
        # ecodes.BTN_TR2: 'R2_click',
    }

    # Common device name fragments for Sony on Linux
    SONY_NAME_HINTS = (
        "Sony Interactive Entertainment Wireless Controller",  # DualSense (PS5)
        "Wireless Controller",                                  # DS4/DualSense generic
        "DUALSHOCK", "DualShock", "DualSense", "PS4 Controller", "PS5 Controller"
    )

    def __init__(self, device_name_hints=SONY_NAME_HINTS, pos_sensitivity=1.0, rot_sensitivity=1.0):
        print("Opening PlayStationGamepad device")
        self.device_path = None
        for dev_path in list_devices():
            dev = InputDevice(dev_path)
            if any(hint in dev.name for hint in device_name_hints):
                self.device_path = dev_path
                break

        if self.device_path is None:
            # Fallback: show available devices to help debugging
            avail = [InputDevice(p).name for p in list_devices()]
            raise RuntimeError(f"No PlayStation controller found. Devices: {avail}")

        self.device = InputDevice(self.device_path)
        print(f"Connected to {self.device.name} at {self.device_path}")

        self.pos_sensitivity = pos_sensitivity
        self.rot_sensitivity = rot_sensitivity

        # 6-DOF variables
        self.x = self.y = self.z = 0.0
        self.roll = self.pitch = self.yaw = 0.0

        self._display_controls()
        self.single_click_and_hold = False

        self._control = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        self._reset_state = 0
        self.rotation = np.array([[-1.0, 0.0, 0.0],
                                  [ 0.0, 1.0, 0.0],
                                  [ 0.0, 0.0,-1.0]])

        self._enabled = False
        self._running = True
        self._collecting = False

        self.buttons = {name: 0 for name in self.BUTTON_MAP.values()}

        self.thread = threading.Thread(target=self.run, daemon=False)
        self.thread.start()

        self.abs_range = {}
        for code in [ecodes.ABS_X, ecodes.ABS_Y, ecodes.ABS_RX, ecodes.ABS_RY,
                    ecodes.ABS_Z, ecodes.ABS_RZ, ecodes.ABS_HAT0X, ecodes.ABS_HAT0Y]:
            try:
                ai = self.device.absinfo(code)
                self.abs_range[code] = (ai.min, ai.max)
            except Exception:
                pass


    @staticmethod
    def _display_controls():
        def print_command(char, info):
            char = (char + " " * 30)[:30]
            print(f"{char}\t{info}")

        print("")
        print_command("Control", "Command")
        print_command("Triangle (press)", "open gripper")
        print_command("Cross (press)", "close gripper")
        print_command("Share button", "reset simulation")
        print_command("DPad (left/right)", "move arm in x direction")
        print_command("DPad (up/down)", "move arm in y direction")
        print_command("Square (press)", "start collecting")
        print_command("Circle (press)", "stop collecting")
        print_command("ESC", "quit")
        print("")

    def _reset_internal_state(self):
        self.rotation = np.array([[-1.0, 0.0, 0.0],
                                  [ 0.0, 1.0, 0.0],
                                  [ 0.0, 0.0,-1.0]])
        self.x = self.y = self.z = 0.0
        self.roll = self.pitch = self.yaw = 0.0
        self._control = np.zeros(6)
        self.single_click_and_hold = False

    def start_control(self):
        self._reset_internal_state()
        self._reset_state = 0
        self._enabled = True
        self._collecting = False

    def stop_control(self):
        self._enabled = False
        self._running = False

    def close(self):
        self.stop_control()
        if hasattr(self, 'thread') and self.thread.is_alive():
            self.thread.join(timeout=1.0)

    def get_controller_state(self):
        dpos = self.control[:3] * 0.005 * self.pos_sensitivity
        roll, pitch, yaw = self.control[3:] * 0.005 * self.rot_sensitivity

        drot1 = rotation_matrix(angle=-pitch, direction=[1.0, 0, 0], point=None)[:3, :3]
        drot2 = rotation_matrix(angle= roll, direction=[0, 1.0, 0], point=None)[:3, :3]
        drot3 = rotation_matrix(angle= yaw, direction=[0, 0, 1.0], point=None)[:3, :3]
        self.rotation = self.rotation.dot(drot1.dot(drot2.dot(drot3)))

        return dict(
            dpos=dpos,
            rotation=self.rotation,
            raw_drotation=np.array([roll, pitch, yaw]),
            grasp=self.control_gripper,
            reset=self._reset_state,
        )

    def run(self):
        try:
            while self._running:
                try:
                    for event in self.device.read_loop():
                        if not self._enabled:
                            time.sleep(0.01)
                            continue
                        if event.type == ecodes.EV_ABS and event.code in self.AXIS_MAP:
                            name = self.AXIS_MAP[event.code]
                            val = event.value
                            amin, amax = self.abs_range.get(event.code, (0, 255))  # 回退值只是兜底
                            # Left stick -> translation X/Y (like your original DPad mapping),
                            # D-Pad still supported for coarse steps.
                            if name == 'LX':
                                # DS axes are centered ~128. Invert X as in your code.
                                self.x = scale_to_control(val, amin, amax) * -1.0
                            elif name == 'LY':
                                # Up is usually negative on LY; keep your sign convention.
                                self.y = scale_to_control(val, amin, amax)
                            elif name == 'L2':
                                # Map L2 analog to Z translation (push to move down).
                                self.z = -scale_trigger(val, amin, amax)
                            elif name == 'R2':
                                # Map R2 analog to yaw (twist).
                                self.yaw = scale_trigger(val, amin, amax)
                            elif name == 'RX':
                                self.roll = 0
                            elif name == 'RY':
                                self.z = scale_centered(val, amin, amax) * -1.0
                            elif name == 'DPad X':
                                # Discrete nudge on X; zero out roll to avoid mixing
                                self.roll = 0
                                # self.x = float(val)  # -1,0,+1
                            elif name == 'DPad Y':
                                self.pitch = 0
                                # self.y = float(-val)  # up=-1 => +Y

                            self._control = [self.x, self.y, self.z, self.roll, self.pitch, self.yaw]

                        elif event.type == ecodes.EV_KEY and event.code in self.BUTTON_MAP:
                            btn_name = self.BUTTON_MAP[event.code]
                            self.buttons[btn_name] = event.value

                            # Gripper logic: Cross closes, Triangle opens
                            if btn_name == 'Cross' and event.value == 1:
                                self.single_click_and_hold = True
                            elif btn_name == 'Triangle' and event.value == 1:
                                self.single_click_and_hold = False

                            # Reset logic
                            if btn_name == 'Share' and event.value == 1:  # share is on the left of touchpad
                                print("[Pad] RESET triggered by Share/Create")
                                self._reset_state = 1
                                self._enabled = False
                                self._reset_internal_state()

                            # Start/stop collecting (Square/Circle)
                            if btn_name == 'Square' and event.value == 1:
                                self._collecting = True
                            if btn_name == 'Circle' and event.value == 1:
                                self._collecting = False

                except Exception as e:
                    if self._running:
                        print(f"[PlayStationGamepad] Exception in run: {e}")
                    break
        except Exception as e:
            print(f"PlayStationGamepad thread error: {e}")
        finally:
            try:
                if hasattr(self, 'device'):
                    self.device.close()
            except Exception:
                pass

    @property
    def control(self):
        return np.array(self._control, dtype=float)

    @property
    def control_gripper(self):
        return 1.0 if self.single_click_and_hold else 0.0

    def __del__(self):
        self.close()
        print("PlayStationGamepad closed.")


if __name__ == "__main__":
    pad = ZikwayGamepad()
    pad.start_control()
    try:
        while True:
            print(pad.control, pad.control_gripper)
            time.sleep(0.02)
    except KeyboardInterrupt:
        print("Interrupted by user")
    finally:
        pad.close()
