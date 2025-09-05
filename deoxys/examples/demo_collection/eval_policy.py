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
from deoxys.utils.io_devices import ZikwayGamepad
from deoxys.utils.cam_utils import load_camera_config 
from deoxys.experimental.motion_utils import reset_joints_to
import torch
import json
from robomimic.config import config_factory
from robomimic.algo import algo_factory, RolloutPolicy
from robomimic.utils.file_utils import maybe_dict_from_checkpoint

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--interface-cfg", type=str, default="franka_gyh.yml")
    parser.add_argument(
        "--controller-cfg", type=str, default="osc-position-controller.yml"
    )
    args = parser.parse_args()
    return args

def main():

    with open('<your_log_dir>/../config.json', 'r') as f:
        ext_cfg = json.load(f)
        config = config_factory(ext_cfg["algo_name"])
    with config.values_unlocked(): 
        config.update(ext_cfg)
    config.lock()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    obs_key_shapes = ext_cfg["shape_meta"]["all_shapes"] 
    ac_dim = ext_cfg["shape_meta"]["ac_dim"]

    model = algo_factory(
        algo_name=config.algo_name,
        config=config,
        obs_key_shapes=obs_key_shapes,
        ac_dim=ac_dim,
        device=device)
    ckpt = maybe_dict_from_checkpoint('<path_to>/model_epoch_XX.pth') 
    model.deserialize(ckpt["model"]) 
    model.eval()

    obs_norm = ckpt.get("obs_normalization_stats", None)
    act_norm = ckpt.get("action_normalization_stats", None)
    policy = RolloutPolicy(model, obs_normalization_stats=obs_norm, action_normalization_stats=act_norm)
    
    args = parse_args()

    ## Initialize Robot Interface 
    print("starting franka interface ...")
    robot_interface = FrankaInterface(config_root + f"/{args.interface_cfg}")

    ## Setup controller type
    controller_cfg = YamlConfig(config_root + f"/{args.controller_cfg}").as_easydict()
    controller_type = "OSC_POSITION"

    ## Move to Initial joint position
    ## This can be modified, if starting point needs to be changed
    joint_start = [0, -np.pi / 4, 0, -3 * np.pi / 4, 0, np.pi / 2, 0]
    print("move to starting point of the trajectory ...")
    print(joint_start)
    reset_joints_to(robot_interface, joint_start)
    time.sleep(1)

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
        ## !! Add one more state of device for checking BUTTON 'A'
        if not collecting and device._collecting:  # assuming button 1 starts the collection
            collecting = True
            print("Started collecting data...")
            start_time = time.time_ns()

        # Stop collection on a specific button press (e.g., button 2 for stop)
        ## !! Add one more state of device for checking BUTTON 'B'
        if collecting and not device._collecting:  # assuming button 2 stops the collection
            collecting = False
            print("Stopped collecting data.")
            break
        

        # if collecting:

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

         # Capture camera images info
        for cam_name, image_content in client.img_contents.items():
            data[cam_name].append(image_content['img_array'])

        end_time = time.time_ns()
        print(f"Time profile: {(end_time - start_time) / 10 ** 9}")

    # client._close()
    robot_interface.close()

if __name__ == "__main__":
    main()