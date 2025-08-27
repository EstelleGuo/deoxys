"""Moving robot joint positions to initial pose for starting new experiments."""

import argparse
import pickle
import threading
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from deoxys import config_root
from deoxys.franka_interface import FrankaInterface
from deoxys.utils import YamlConfig
from deoxys.utils.input_utils import input2action
from deoxys.utils.io_devices import SpaceMouse
from deoxys.utils.log_utils import get_deoxys_example_logger

logger = get_deoxys_example_logger()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--interface-cfg", type=str, default="franka_right.yml")
    parser.add_argument(
        "--controller-cfg", type=str, default="joint-position-controller.yml"
    )
    parser.add_argument(
        "--folder", type=Path, default="data_collection_example/example_data"
    )

    args = parser.parse_args()
    return args

class Robot(FrankaInterface):
    def __init__(self, control_freq):
        super(Robot, self).__init__(
            general_cfg_file= "/home/gn/Documents/Reaserach_Space/nyu_learning/deoxys_control/deoxys/config/franka_right.yml",
            use_visualizer=False,
            control_freq=control_freq,
        )

    def reset_robot(self):
        self.reset()

        print("Waiting for the robot to connect...")
        while len(self._state_buffer) == 0:
            time.sleep(0.01)

        print("Franka is connected")


def main():
    # args = parse_args()

    # robot_interface = FrankaInterface(
    #     config_root + f"/{args.interface_cfg}", use_visualizer=True
    # )

    # robot_interface.reset()

    # print("Waiting for the robot to connect...")
    # while len(robot_interface._state_buffer) == 0:
    #     time.sleep(0.01)

    # print("Franka is connected")

    # robot_interface.close()

    t = Robot(control_freq=20)
    t.reset_robot()


if __name__ == "__main__":
    main()
