import argparse
import os
import pickle
import threading
import time
from pathlib import Path
from dataclasses_json import config
import matplotlib.pyplot as plt
import numpy as np
from deoxys import config_root
from deoxys.sensor_interface.camera_client import CameraClient
from deoxys.franka_interface import FrankaInterface
from deoxys.utils import YamlConfig
from deoxys.utils.input_utils import input2action
from deoxys.utils.io_devices import SpaceMouse
from deoxys.utils.io_devices import ZikwayGamepad, DualSenseGamepad
from deoxys.utils.cam_utils import load_camera_config 
from deoxys.experimental.motion_utils import reset_joints_to

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--interface-cfg", type=str, default="franka_gn.yml")
    parser.add_argument(
        "--controller-cfg", type=str, default="osc-position-controller.yml"
    )
    parser.add_argument("--folder", type=Path, default="example_data")
    args = parser.parse_args()
    return args

def main():
    args = parse_args()

    args.folder.mkdir(parents=True, exist_ok=True)

    experiment_id = 0
    for path in args.folder.glob("run*"):
        if not path.is_dir():
            continue
        try:
            folder_id = int(str(path).split("run")[-1])
            if folder_id > experiment_id:
                experiment_id = folder_id
        except BaseException:
            pass
    experiment_id += 1
    folder = str(args.folder / f"run{experiment_id}")
    os.makedirs(folder, exist_ok=True)

    ## Initialize Robot Interface 
    print("starting franka interface ...")
    robot_interface = FrankaInterface(config_root + f"/{args.interface_cfg}")

    ## Setup controller type
    controller_cfg = YamlConfig(config_root + f"/{args.controller_cfg}").as_easydict()
    controller_type = "OSC_POSITION"

    ## Move to Initial joint position
    ## This can be modified, if starting point needs to be changed
    joint_start = [0., -np.pi / 4, 0., -3 * np.pi / 4, 0., np.pi / 2, 0.]
    print("move to starting point of the trajectory ...")
    print(joint_start)
    reset_joints_to(robot_interface, joint_start)
    time.sleep(1)

    print("starting gamepad controller ...")
    device = DualSenseGamepad(pos_sensitivity= 1.0, rot_sensitivity=1.0)
    device.start_control()

    # Initialize camera interfaces. 
    print("starting camera sensors ...")
    config = load_camera_config()   
    camera_infos = config["camera_infos"]
    client = CameraClient(config, image_show = True, Unit_Test=False) # local test
    image_receive_thread = threading.Thread(target = client.receive_process, daemon = True)
    image_receive_thread.daemon = True
    image_receive_thread.start()

    data = {
        "action": [],
        "proprio_ee": [],
        "proprio_joints": [],
        "proprio_gripper_state": [],
        "robot_eef_pose": [],  #  -------->   proprio_ee
        "robot_eef_pose_vel": [], # ---------> this needs to be complete after data collection J(q) @ dq
        "robot_joint": [], # --------->  proprio_joints
        "robot_joint_vel": [], # ---------->   data["robot_joint_vel"].append(np.array(last_state.dq))
        "stage": [],
        "timestamp": []
    }

    for cam_info in camera_infos:
        data[cam_info.camera_name] = []
    
    i = 0
    start = False
    start_time = None
    collecting = False
    # time.sleep(2)  # wait a bit for everything to start
    print("start---------------------------------------------")
    while True:
        start_time = time.time_ns()
        action, grasp = input2action(
            device=device,
            controller_type=controller_type,
        )
        
        if action is None:
            break

        # Start collection on a specific button press (e.g., button 1 for start)
        ## !! Add one more state of device for checking BUTTON 'Cross'
        if not collecting and device._collecting:  # assuming button 1 starts the collection
            collecting = True
            print("Started collecting data...")
            start_time = time.time_ns()

        # Stop collection on a specific button press (e.g., button 2 for stop)
        ## !! Add one more state of device for checking BUTTON 'Circle'
        if collecting and not device._collecting:  # assuming button 2 stops the collection
            collecting = False
            print("Stopped collecting data.")
            break
        
        robot_interface.control(
            controller_type=controller_type,
            action=action,
            controller_cfg=controller_cfg,
        )
        if len(robot_interface._state_buffer) == 0:
            continue

        last_state = robot_interface._state_buffer[-1]
        # print("Gripper state buffer:", robot_interface._gripper_state_buffer)
        if robot_interface._gripper_state_buffer:
            last_gripper_state = robot_interface._gripper_state_buffer[-1]
        else:
            # Handle the empty case appropriately
            last_gripper_state = None

        if np.linalg.norm(action[:-1]) < 1e-3 and not collecting:
            continue

        print(action)

        # Record ee pose,  joints, action, and gripper state(if use gripper)
        data["action"].append(action)
        data["robot_eef_pose"].append(np.array(last_state.O_T_EE))
        data["robot_joint"].append(np.array(last_state.q))

        if last_gripper_state is not None:
            data["proprio_gripper_state"].append(np.array(last_gripper_state.width))
        
        # Get img info
        # Capture camera images info - only RGB/color stream
        for cam_name, image_content in client.img_contents.items():
            if 'color' in image_content['streams']:
                data[cam_name].append(image_content['streams']['color']['img_array'].copy())

        end_time = time.time_ns()
        print(f"Time profile: {(end_time - start_time) / 10 ** 9}")

    np.savez(f"{folder}/testing_demo_action", data=np.array(data["action"]))
    np.savez(f"{folder}/testing_demo_proprio_ee", data=np.array(data["robot_eef_pose"]))
    np.savez(
        f"{folder}/testing_demo_proprio_joints", data=np.array(data["robot_joint"])
    )
    if len(data["proprio_gripper_state"]) > 0:
        np.savez(
            f"{folder}/testing_demo_proprio_gripper_state",
            data=np.array(data["proprio_gripper_state"]),
        )
    else:
        print("No gripper state data collected, skipping save.")

    for cam_info in camera_infos:
        np.savez(f"{folder}/testing_demo_camera_{cam_info.camera_name}", data=np.array(data[cam_info.camera_name]))

    print("Franka Interface Stopped.")
    robot_interface.close()
    print("Gamepad Interface Stopped.")
    device.stop_control()
    print("Camera Interface Stopped.")
    client.stop()
    image_receive_thread.join(timeout=2.0)

    save = input("Save or not? (enter 0 or 1)")
    save = bool(int(save))

    if not save:
        import shutil
        shutil.rmtree(f"{folder}")
        print("Data not saved. Exiting.")
    else:
        print("Data saved. Exiting.")
    import sys
    sys.exit(0)

if __name__ == "__main__":
    main()