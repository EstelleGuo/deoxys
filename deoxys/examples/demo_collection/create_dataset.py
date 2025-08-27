import argparse
import os
from pathlib import Path

import cv2
import h5py
import numpy as np
from deoxys.utils.cam_utils import load_camera_config, resize_img 


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--interface-cfg", type=str, default="franka_gn.yml")
    parser.add_argument("--controller-cfg", type=str, default="osc-position-controller.yml")
    parser.add_argument("--folder", type=Path, default="example_data")

    args = parser.parse_args()
    return args

def main():
    args = parse_args()
    camera_infos = load_camera_config()['camera_infos']
    camera_names = [cam_info.camera_name for cam_info in camera_infos]

    folder = str(args.folder)

    demo_file_name = f"{folder}/demo.hdf5"
    demo_file = h5py.File(demo_file_name, "w")

    grp = demo_file.create_group("data")

    num_demos = 0

    for (run_idx, path) in enumerate(Path(folder).glob("run*")):
        print(run_idx)

        num_demos += 1
        proprio_joints = []
        proprio_ee = []
        proprio_gripper_state = []
        actions = []

        action_data = np.load(f"{path}/testing_demo_action.npz", allow_pickle=True)[
            "data"
        ]
        proprio_ee_data = np.load(
            f"{path}/testing_demo_proprio_ee.npz", allow_pickle=True
        )["data"]
        proprio_joints_data = np.load(
            f"{path}/testing_demo_proprio_joints.npz", allow_pickle=True
        )["data"]

        ####### If we have collected this 
        # proprio_gripper_state_data = np.load(
        #     f"{path}/testing_demo_proprio_gripper_state.npz", allow_pickle=True
        # )["data"]
        proprio_gripper_state_data = np.zeros(len(action_data))

        # We should add this later !!!!
        rewards = np.zeros(len(action_data))

        len_data = len(action_data)

        if len_data == 0:
            print(f"Data incorrect: {run_idx}")
            continue

        demo_grp = grp.create_group(f"demo_{run_idx}")

        camera_data = {}
        for camera_info in camera_infos:
            camera_data[camera_info.camera_name] = np.load(
                f"{path}/testing_demo_camera_{camera_info.camera_name}.npz", allow_pickle=True
            )["data"]


        assert len(proprio_ee_data) == len(action_data)

        image_color_data = {}
        image_depth_data = {}

        for camera_info in camera_infos:
            image_color_data[camera_info.camera_name] = []
            image_depth_data[camera_info.camera_name] = []

        print("Length of data", len_data)

        demo_obs_grp = demo_grp.create_group("obs")
        

        for i in range(len_data):
            for camera_info in camera_infos:
                img = camera_data[camera_info.camera_name][i]
                try:
                    resized_img = resize_img(img)
                except:
                    import pdb

                    pdb.set_trace()
                image_color_data[camera_info.camera_name].append(resized_img)

                ################## If we have depth image to save
                # if "depth_img_name" in camera_data[camera_name][i]:
                #     depth_image_name = camera_data[camera_name][i]["depth_img_name"]
                #     img = cv2.imread(depth_image_name)
                #     # resized_img = cv2.resize(img, (0, 0), fx=0.2, fy=0.2)
                #     resized_img = resize_img(img, camera_type=camera_type)
                #     new_image_name = f"{image_name}_depth.tiff"
                #     cv2.imwrite(new_image_name, resized_img)
                #     image_depth_data[camera_name].append(resized_img.transpose(2, 0, 1))
                #     image_depth_names_data[camera_name].append(new_image_name)

            proprio_ee.append(proprio_ee_data[i])
            proprio_joints.append(proprio_joints_data[i])
            proprio_gripper_state.append([proprio_gripper_state_data[i]])
            actions.append(action_data[i])

        assert len(actions) == len(proprio_ee)
        assert len(image_color_data[camera_infos[0].camera_name]) == len(actions)

        for camera_info in camera_infos:
            demo_data_grp = demo_obs_grp.create_dataset(
                f"{camera_info.camera_name}_color", data=image_color_data[camera_info.camera_name]
            )
            demo_data_grp.attrs["resized_color_img_h"] = image_color_data[camera_info.camera_name][0].shape[0]
            demo_data_grp.attrs["resized_color_img_w"] = image_color_data[camera_info.camera_name][0].shape[1]
            demo_data_grp.attrs["original_color_img_h"] = camera_info.cfg['height']
            demo_data_grp.attrs["original_color_img_w"] = camera_info.cfg['width']

            ################## If we have depth image to save
            # ep_data_grp = demo_grp.create_dataset(f"camera_{camera_id}_depth", data=image_depth_names_data[camera_id])
            # image_color_data_list[camera_name].append(
            #     np.stack(image_color_data[camera_name], axis=0)
            # )
            # image_depth_data_list[camera_id].append(np.stack(image_depth_data[camera_id], axis=0))


        demo_data_grp = demo_obs_grp.create_dataset(
            "proprio_joints", data=np.stack(proprio_joints)
        )
        demo_data_grp = demo_obs_grp.create_dataset("proprio_ee", data=np.stack(proprio_ee))
        demo_data_grp = demo_obs_grp.create_dataset(
            "proprio_gripper_state", data=np.stack(proprio_gripper_state)
        )

        demo_data_grp = demo_grp.create_dataset("actions", data=np.stack(actions))
        
    ###### ADD THIS LATER
    grp.attrs["num_demos"] = num_demos
    grp.attrs["attributes"] = ["joints", "ee", "gripper_state", "actions"]
    grp.attrs["camera_name"] = camera_names


    demo_file.close()


if __name__ == "__main__":
    main()
