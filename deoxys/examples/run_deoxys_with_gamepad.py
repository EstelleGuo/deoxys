# import argparse
# import time
# from deoxys import config_root
# from deoxys.franka_interface import FrankaInterface
# from deoxys.utils.config_utils import get_default_controller_config
# from deoxys.utils.input_utils import input2action
# from deoxys.utils.io_devices import ZikwayGamepad
# from deoxys.utils.log_utils import get_deoxys_example_logger

# logger = get_deoxys_example_logger()

# def main():

#     parser = argparse.ArgumentParser()
#     parser.add_argument("--interface-cfg", type=str, default="franka_gyh.yml")
#     parser.add_argument("--controller-type", type=str, default="OSC_POSE")

#     args = parser.parse_args()
#     device = ZikwayGamepad(pos_sensitivity= 1.0, rot_sensitivity=1.0)
#     device.start_control()

#     robot_interface = FrankaInterface(config_root + f"/{args.interface_cfg}", use_visualizer=True)

#     controller_type = args.controller_type
#     controller_cfg = get_default_controller_config(controller_type=controller_type)

#     robot_interface._state_buffer = []

#     for i in range(3000):
#         start_time = time.time_ns()

#         action, grasp = input2action(
#             device=device,
#             controller_type=controller_type,
#         )

#         robot_interface.control(
#             controller_type=controller_type,
#             action=action,
#             controller_cfg=controller_cfg,
#         )
#         end_time = time.time_ns()
#         logger.debug(f"Time duration: {((end_time - start_time) / (10**9))}")

#     robot_interface.control(
#         controller_type=controller_type,
#         action=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0] + [1.0],
#         controller_cfg=controller_cfg,
#         termination=True,
#     )

#     robot_interface.close()

#     # Check if there is any state frame missing
#     for (state, next_state) in zip(
#         robot_interface._state_buffer[:-1], robot_interface._state_buffer[1:]
#     ):
#         if (next_state.frame - state.frame) > 1:
#             print(state.frame, next_state.frame)


# if __name__ == "__main__":
#     main()

import argparse
import time
import numpy as np
from pathlib import Path

from deoxys import config_root
from deoxys.franka_interface import FrankaInterface
from deoxys.utils import YamlConfig
from deoxys.utils.input_utils import input2action
from deoxys.utils.io_devices import ZikwayGamepad
from deoxys.utils.config_utils import get_default_controller_config
from deoxys.utils.log_utils import get_deoxys_example_logger

logger = get_deoxys_example_logger()

def reset_robot_joints(robot_interface, controller_cfg):
    # Define the reset joint positions
    reset_joint_positions = [0, -np.pi/4, 0, -3/4 * np.pi, 0, np.pi/2, np.pi/4]
    action = reset_joint_positions + [1.0]

    # Move the robot to the reset joint positions
    while True:
        if len(robot_interface._state_buffer) > 0:
            logger.info(f"Current Robot joint: {np.round(robot_interface.last_q, 3)}")
            logger.info(f"Desired Robot joint: {np.round(robot_interface.last_q_d, 3)}")

            if np.max(np.abs(np.array(robot_interface._state_buffer[-1].q) - np.array(reset_joint_positions))) < 1e-3:
                break
        robot_interface.control(controller_type="JOINT_POSITION", action=action, controller_cfg=controller_cfg)
    logger.info("Robot joints are reset to initial position.")

def gamepad_control(robot_interface, device, controller_type, controller_cfg):
    # Start gamepad control and robot control
    for i in range(3000):
        start_time = time.time_ns()

        # Get the action from the gamepad
        action, grasp = input2action(device=device, controller_type=controller_type)

        # Send the action to control the robot
        robot_interface.control(controller_type=controller_type, action=action, controller_cfg=controller_cfg)
        
        end_time = time.time_ns()
        logger.debug(f"Time duration: {((end_time - start_time) / (10**9))}")

    # Send termination action to stop robot movement
    robot_interface.control(controller_type=controller_type, action=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0] + [1.0], controller_cfg=controller_cfg, termination=True)
    logger.info("Gamepad control finished.")

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--interface-cfg", type=str, default="franka_gyh.yml")
    parser.add_argument("--controller-type", type=str, default="OSC_POSE")
    parser.add_argument("--controller-cfg", type=str, default="joint-position-controller.yml")
    args = parser.parse_args()
    return args

def main():
    args = parse_args()

    # Initialize robot interface
    robot_interface = FrankaInterface(config_root + f"/{args.interface_cfg}", use_visualizer=True)
    controller_cfg = YamlConfig(config_root + f"/{args.controller_cfg}").as_easydict()

    # Reset the robot joints to initial position
    reset_robot_joints(robot_interface, controller_cfg)

    # Initialize gamepad
    device = ZikwayGamepad(pos_sensitivity=1.0, rot_sensitivity=1.0)
    device.start_control()

    # Get the controller configuration based on the controller type
    controller_type = args.controller_type
    controller_cfg = get_default_controller_config(controller_type=controller_type)

    # Start gamepad control
    gamepad_control(robot_interface, device, controller_type, controller_cfg)

    # Close the robot interface
    robot_interface.close()

    # Check if there are any missing state frames
    for (state, next_state) in zip(robot_interface._state_buffer[:-1], robot_interface._state_buffer[1:]):
        if (next_state.frame - state.frame) > 1:
            print(f"Frame missing: {state.frame} -> {next_state.frame}")

if __name__ == "__main__":
    main()

