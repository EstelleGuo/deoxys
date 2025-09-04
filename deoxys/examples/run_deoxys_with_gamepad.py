import argparse
import time
import numpy as np
from deoxys import config_root
from deoxys.franka_interface import FrankaInterface
from deoxys.utils.config_utils import get_default_controller_config
from deoxys.utils.input_utils import input2action
from deoxys.utils.io_devices import ZikwayGamepad, DualSenseGamepad
from deoxys.utils.log_utils import get_deoxys_example_logger
from deoxys.experimental.motion_utils import reset_joints_to

logger = get_deoxys_example_logger()

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--interface-cfg", type=str, default="franka_gn.yml")
    parser.add_argument("--controller-type", type=str, default="OSC_POSITION")

    args = parser.parse_args()
    device = DualSenseGamepad(pos_sensitivity= 1.0, rot_sensitivity=1.0)
    device.start_control()

    robot_interface = FrankaInterface(config_root + f"/{args.interface_cfg}", use_visualizer=True)

    controller_type = args.controller_type
    controller_cfg = get_default_controller_config(controller_type=controller_type)

    robot_interface._state_buffer = []


    ## Move to Initial joint position
    ## This can be modified, if starting point needs to be changed
    joint_start = [0, -np.pi / 4, 0, -3 * np.pi / 4, 0, np.pi / 2, 0]
    print("move to starting point of the trajectory ...")
    print(joint_start)
    reset_joints_to(robot_interface, joint_start)

    
    for i in range(1000):
        start_time = time.time_ns()

        action, grasp = input2action(
            device=device,
            controller_type=controller_type,
        )

        robot_interface.control(
            controller_type=controller_type,
            action=action,
            controller_cfg=controller_cfg,
        )
        end_time = time.time_ns()
        logger.debug(f"Time duration: {((end_time - start_time) / (10**9))}")

    robot_interface.control(
        controller_type=controller_type,
        action=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0] + [1.0],
        controller_cfg=controller_cfg,
        termination=True,
    )

    robot_interface.close()

    # Check if there is any state frame missing
    for (state, next_state) in zip(
        robot_interface._state_buffer[:-1], robot_interface._state_buffer[1:]
    ):
        if (next_state.frame - state.frame) > 1:
            print(state.frame, next_state.frame)


if __name__ == "__main__":
    main()
