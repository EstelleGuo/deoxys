# import argparse
# import os
# from pathlib import Path

# import cv2
# import h5py
# import numpy as np
# from deoxys.utils.cam_utils import load_camera_config, resize_img 
# import json

# def parse_args():
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--interface-cfg", type=str, default="franka_gyh.yml")
#     parser.add_argument("--controller-cfg", type=str, default="osc-position-controller.yml")
#     parser.add_argument("--folder", type=Path, default="example_data")

#     args = parser.parse_args()
#     return args

# def main():
#     args = parse_args()
#     camera_infos = load_camera_config()['camera_infos']
#     camera_names = [cam_info.camera_name for cam_info in camera_infos]

#     folder = str(args.folder)

#     demo_file_name = f"{folder}/demo.hdf5"
#     demo_file = h5py.File(demo_file_name, "w")

#     grp = demo_file.create_group("data")

#     num_demos = 0

#     for (run_idx, path) in enumerate(Path(folder).glob("run*")):
#         print(run_idx, path)

#         num_demos += 1
#         proprio_joints = []
#         proprio_ee = []
#         proprio_gripper_state = []
#         actions = []

#         action_data = np.load(f"{path}/testing_demo_action.npz", allow_pickle=True)[
#             "data"
#         ]
#         proprio_ee_data = np.load(
#             f"{path}/testing_demo_proprio_ee.npz", allow_pickle=True
#         )["data"]
#         proprio_joints_data = np.load(
#             f"{path}/testing_demo_proprio_joints.npz", allow_pickle=True
#         )["data"]

#         ####### If we have collected this 
#         # proprio_gripper_state_data = np.load(
#         #     f"{path}/testing_demo_proprio_gripper_state.npz", allow_pickle=True
#         # )["data"]
#         proprio_gripper_state_data = np.zeros(len(action_data))

#         # We should add this later !!!!
#         rewards = np.zeros(len(action_data))

#         len_data = len(action_data)

#         if len_data == 0:
#             print(f"Data incorrect: {run_idx}")
#             continue

#         demo_grp = grp.create_group(f"demo_{run_idx}")

#         camera_data = {}
#         for camera_info in camera_infos:
#             camera_data[camera_info.camera_name] = np.load(
#                 f"{path}/testing_demo_camera_{camera_info.camera_name}.npz", allow_pickle=True
#             )["data"]


#         assert len(proprio_ee_data) == len(action_data)

#         image_color_data = {}
#         image_depth_data = {}

#         for camera_info in camera_infos:
#             image_color_data[camera_info.camera_name] = []
#             image_depth_data[camera_info.camera_name] = []

#         print("Length of data", len_data)

#         demo_obs_grp = demo_grp.create_group("obs")
        

#         for i in range(len_data):
#             for camera_info in camera_infos:
#                 img = camera_data[camera_info.camera_name][i]
#                 try:
#                     resized_img = resize_img(img)
#                 except:
#                     import pdb

#                     pdb.set_trace()
#                 image_color_data[camera_info.camera_name].append(resized_img)

#                 ################## If we have depth image to save
#                 # if "depth_img_name" in camera_data[camera_name][i]:
#                 #     depth_image_name = camera_data[camera_name][i]["depth_img_name"]
#                 #     img = cv2.imread(depth_image_name)
#                 #     # resized_img = cv2.resize(img, (0, 0), fx=0.2, fy=0.2)
#                 #     resized_img = resize_img(img, camera_type=camera_type)
#                 #     new_image_name = f"{image_name}_depth.tiff"
#                 #     cv2.imwrite(new_image_name, resized_img)
#                 #     image_depth_data[camera_name].append(resized_img.transpose(2, 0, 1))
#                 #     image_depth_names_data[camera_name].append(new_image_name)

#             proprio_ee.append(proprio_ee_data[i])
#             proprio_joints.append(proprio_joints_data[i])
#             proprio_gripper_state.append([proprio_gripper_state_data[i]])
#             actions.append(action_data[i])

#         assert len(actions) == len(proprio_ee)
#         assert len(image_color_data[camera_infos[0].camera_name]) == len(actions)

#         for camera_info in camera_infos:
#             demo_data_grp = demo_obs_grp.create_dataset(
#                 f"{camera_info.camera_name}_color", data=image_color_data[camera_info.camera_name]
#             )
#             demo_data_grp.attrs["resized_color_img_h"] = image_color_data[camera_info.camera_name][0].shape[0]
#             demo_data_grp.attrs["resized_color_img_w"] = image_color_data[camera_info.camera_name][0].shape[1]
#             demo_data_grp.attrs["original_color_img_h"] = camera_info.cfg['height']
#             demo_data_grp.attrs["original_color_img_w"] = camera_info.cfg['width']

#             ################## If we have depth image to save
#             # ep_data_grp = demo_grp.create_dataset(f"camera_{camera_id}_depth", data=image_depth_names_data[camera_id])
#             # image_color_data_list[camera_name].append(
#             #     np.stack(image_color_data[camera_name], axis=0)
#             # )
#             # image_depth_data_list[camera_id].append(np.stack(image_depth_data[camera_id], axis=0))


#         demo_data_grp = demo_obs_grp.create_dataset(
#             "proprio_joints", data=np.stack(proprio_joints)
#         )
#         demo_data_grp = demo_obs_grp.create_dataset("proprio_ee", data=np.stack(proprio_ee))
#         demo_data_grp = demo_obs_grp.create_dataset(
#             "proprio_gripper_state", data=np.stack(proprio_gripper_state)
#         )

#         demo_data_grp = demo_grp.create_dataset("actions", data=np.stack(actions))
        
#     ###### ADD THIS LATER
#     grp.attrs["num_demos"] = num_demos
#     grp.attrs["attributes"] = ["joints", "ee", "gripper_state", "actions"]
#     grp.attrs["camera_name"] = camera_names
#     grp.attrs["env_args"] = json.dumps({
#                 "env_name": "PushT-Real",
#                 "env_version": "1.0.0",
#                 # Keep the same schema: an int "type" field.
#                 # If your downstream code *expects* robosuite=1, you can keep 1.
#                 # Otherwise, you can choose another code (e.g., 2) to mark "real".
#                 "type": 2,
#                 "env_kwargs": {
#                     # Real world: no sim rendering
#                     "has_renderer": False,
#                     "has_offscreen_renderer": False,

#                     # In real data collection we usually don't terminate episodes early
#                     "ignore_done": True,

#                     # We typically don't have perfect object-state observations in real setups
#                     "use_object_obs": False,

#                     # We *do* use cameras; your dataset already stores images
#                     "use_camera_obs": True,

#                     # Control/logging frequency used during collection (adjust if yours differs)
#                     "control_freq": 20,

#                     # Real world: controller configs are not needed by robomimic for *offline* training,
#                     # but we keep the key for parity (set to empty dict).
#                     "controller_configs": {},

#                     # Physical robot(s) used
#                     "robots": ["Franka"],

#                     # Camera image sizes used in the dataset; pick what you actually wrote.
#                     "camera_depths": False,
#                     "camera_heights": 480,
#                     "camera_widths": 640,

#                     # No simulator physics flags in real data
#                     "lite_physics": False,

#                     # Many real datasets use sparse success signals; keep shaping off
#                     "reward_shaping": False,

#                     # Optional: episode horizon used during logging/segmentation
#                     "horizon": 200,

#                     # Optional notes to make intent clear
#                     "notes": "Real-world PushT dataset collected with Deoxys; images and proprio only."
#                 }
#             })

#     grp.attrs["total"] = num_demos


#     demo_file.close()


# if __name__ == "__main__":
#     main()

import argparse
from pathlib import Path
import json

import h5py
import numpy as np
from deoxys.utils.cam_utils import load_camera_config, resize_img  # cv2 not needed here


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--interface-cfg", type=str, default="franka_gyh.yml")
    parser.add_argument("--controller-cfg", type=str, default="osc-position-controller.yml")
    parser.add_argument("--folder", type=Path, default=Path("example_data"))
    return parser.parse_args()


def main():
    args = parse_args()
    out_dir = Path(args.folder).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    camera_infos = load_camera_config()['camera_infos']
    camera_names = [ci.camera_name for ci in camera_infos]

    demo_file_path = out_dir / "pusht.hdf5"

    # string dtype for list-like attributes
    str_dtype = h5py.string_dtype(encoding="utf-8")

    num_demos = 0

    with h5py.File(str(demo_file_path), "w") as f:
        grp = f.create_group("data")

        # run_path = Path("/home/hinton/deoxys_gyh/example_data/run43")

        # # load npz chunks
        # actions_npz = np.load(run_path / "testing_demo_action.npz", allow_pickle=True)
        # ee_npz      = np.load(run_path / "testing_demo_proprio_ee.npz", allow_pickle=True)
        # joints_npz  = np.load(run_path / "testing_demo_proprio_joints.npz", allow_pickle=True)

        # actions_arr = actions_npz["data"]
        # proprio_ee_arr = ee_npz["data"]
        # proprio_joints_arr = joints_npz["data"]
        # T = len(actions_arr)
        # # synthetic gripper state (shape [T, 1])
        # proprio_gripper_state_arr = np.zeros((T, 1), dtype=np.float32)

        # # sanity
        # assert len(proprio_ee_arr) == T and len(proprio_joints_arr) == T

        # # load cameras
        # cam_arrays = {}
        # for ci in camera_infos:
        #     cam_npz = np.load(run_path / f"testing_demo_camera_{ci.camera_name}.npz", allow_pickle=True)
        #     frames = cam_npz["data"]  # list/array length T
        #     # resize & stack to (T, H, W, C)
        #     resized = [resize_img(frames[i]) for i in range(T)]
        #     cam_arrays[ci.camera_name] = np.stack(resized, axis=0)

        # # create demo group
        # demo_grp = grp.create_group(f"demo_43")
        # obs_grp = demo_grp.create_group("obs")

        # # write images
        # for ci in camera_infos:
        #     key = f"{ci.camera_name}_color"
        #     arr = cam_arrays[ci.camera_name]
        #     dset = obs_grp.create_dataset(key, data=arr, compression="gzip", compression_opts=4)
        #     # record sizes
        #     dset.attrs["resized_color_img_h"] = int(arr.shape[1])  # H
        #     dset.attrs["resized_color_img_w"] = int(arr.shape[2])  # W
        #     dset.attrs["original_color_img_h"] = int(ci.cfg['height'])
        #     dset.attrs["original_color_img_w"] = int(ci.cfg['width'])

        # # write low-dim obs
        # obs_grp.create_dataset("proprio_joints", data=np.asarray(proprio_joints_arr))
        # obs_grp.create_dataset("proprio_ee", data=np.asarray(proprio_ee_arr))
        # obs_grp.create_dataset("proprio_gripper_state", data=np.asarray(proprio_gripper_state_arr))

        # # actions
        # demo_grp.create_dataset("actions", data=np.asarray(actions_arr))


        # num_demos += 1
        # iterate runs
        for run_idx, run_path in enumerate(sorted([out_dir / f"run{i}" for i in range(44, 75)])):
            run_path = run_path.resolve()
            if not run_path.is_dir():
                continue
            print(run_idx, run_path)

            # load npz chunks
            actions_npz = np.load(run_path / "testing_demo_action.npz", allow_pickle=True)
            ee_npz      = np.load(run_path / "testing_demo_proprio_ee.npz", allow_pickle=True)
            joints_npz  = np.load(run_path / "testing_demo_proprio_joints.npz", allow_pickle=True)

            actions_arr = actions_npz["data"]
            proprio_ee_arr = ee_npz["data"]
            proprio_joints_arr = joints_npz["data"]

            T = len(actions_arr)
            if T == 0:
                print(f"[skip] empty data: {run_idx}")
                continue

            # synthetic gripper state (shape [T, 1])
            proprio_gripper_state_arr = np.zeros((T, 1), dtype=np.float32)

            # sanity
            assert len(proprio_ee_arr) == T and len(proprio_joints_arr) == T

            # load cameras
            cam_arrays = {}
            for ci in camera_infos:
                cam_npz = np.load(run_path / f"testing_demo_camera_{ci.camera_name}.npz", allow_pickle=True)
                frames = cam_npz["data"]  # list/array length T
                # resize & stack to (T, H, W, C)
                resized = [resize_img(frames[i]) for i in range(T)]
                cam_arrays[ci.camera_name] = np.stack(resized, axis=0)

            # create demo group
            demo_grp = grp.create_group(f"demo_{run_idx}")
            obs_grp = demo_grp.create_group("obs")

            # write images
            for ci in camera_infos:
                key = f"{ci.camera_name}_color"
                arr = cam_arrays[ci.camera_name]
                dset = obs_grp.create_dataset(key, data=arr, compression="gzip", compression_opts=4)
                # record sizes
                dset.attrs["resized_color_img_h"] = int(arr.shape[1])  # H
                dset.attrs["resized_color_img_w"] = int(arr.shape[2])  # W
                dset.attrs["original_color_img_h"] = int(ci.cfg['height'])
                dset.attrs["original_color_img_w"] = int(ci.cfg['width'])

            # write low-dim obs
            obs_grp.create_dataset("proprio_joints", data=np.asarray(proprio_joints_arr))
            obs_grp.create_dataset("proprio_ee", data=np.asarray(proprio_ee_arr))
            obs_grp.create_dataset("proprio_gripper_state", data=np.asarray(proprio_gripper_state_arr))

            # actions
            demo_grp.create_dataset("actions", data=np.asarray(actions_arr))

            # required by robomimic SequenceDataset
            demo_grp.attrs["num_samples"] = np.int64(T)

            num_demos += 1

        # dataset-level attrs
        grp.attrs["num_demos"]  = np.int64(num_demos)
        grp.attrs["total"]      = np.int64(num_demos)
        grp.attrs["camera_name"] = np.array(camera_names, dtype=str_dtype)
        grp.attrs["attributes"]  = np.array(["joints", "ee", "gripper_state", "actions"], dtype=str_dtype)

        # env_args as JSON string
        env_args = {
            "env_name": "PushT-Real",
            "env_version": "1.0.0",
            "type": 2,  # distinguish from sim if your stack checks this
            "env_kwargs": {
                "has_renderer": False,
                "has_offscreen_renderer": False,
                "ignore_done": True,
                "use_object_obs": False,
                "use_camera_obs": True,
                "control_freq": 20,
                "controller_configs": {},
                "robots": ["Franka"],
                # your raw camera capture size; obs datasets hold resized frames
                "camera_depths": False,
                "camera_heights": 480,
                "camera_widths": 640,
                "lite_physics": False,
                "reward_shaping": False,
                "horizon": 200,
                "notes": "Real-world PushT dataset collected with Deoxys; images and proprio only."
            }
        }
        grp.attrs["env_args"] = json.dumps(env_args)

    print(f"Done. Wrote {num_demos} demos to {demo_file_path}")


if __name__ == "__main__":
    main()

