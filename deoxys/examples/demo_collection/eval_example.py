# real_inference.py
import os
import time
import json
import numpy as np
import torch
import cv2

from robomimic.algo import algo_factory, RolloutPolicy
from robomimic.config import config_factory
from robomimic.utils import obs_utils as ObsUtils
from robomimic.utils import torch_utils as TorchUtils
from robomimic.utils import file_utils as FileUtils

# === 你自己的传感器 / 机器人接口 ===
from your_camera_reader import get_rgb_frame  # -> np.ndarray HxWx3 uint8 (BGR/RGB看你实现)
from your_robot_io import (
    get_joint_positions,          # -> np.ndarray [n_joints]
    get_eef_pose,                 # -> np.ndarray [7] (pos[3] + quat[4]) 任选其一，和训练时一致
    get_gripper_state,            # -> float or small vec
    send_cartesian_delta_command, # -> 下发末端Δ位姿命令（例）
    send_joint_velocity_command,  # -> 或者关节速度命令（例）
    send_gripper_command,         # -> 夹爪开合
    robot_is_ok,                  # -> bool
)

"""
使用方法：
python real_inference.py --ckpt ckpts/last.pth --fps 10

注意：
1) obs键名必须和训练一致（从ckpt里读 shape_meta/all_obs_keys）
2) 你需要把 policy 输出的动作映射到真机控制（下面给了两个常见示例）
"""

import argparse

def load_from_ckpt(ckpt_path, device):
    # 读 checkpoint（包含：model、config、env_meta、shape_meta、(obs/action)norm stats）
    ckpt = FileUtils.load_dict_from_checkpoint(ckpt_path)
    cfg_dict = ckpt["config"]
    algo_name = cfg_dict["algo_name"]
    # 构造 config（用与训练一致的 algo 基础 config，再更新）
    config = config_factory(algo_name)
    with config.values_unlocked():
        config.update(cfg_dict)
    config.lock()

    # 初始化 ObsUtils（决定图像通道、归一化预处理等）
    ObsUtils.initialize_obs_utils_with_config(config)

    # 从 ckpt 直接拿 obs/action 维度信息
    shape_meta = ckpt["shape_meta"] if "shape_meta" in ckpt else ckpt["model"]["shape_meta"]
    obs_key_shapes = shape_meta["all_shapes"]
    ac_dim = shape_meta["ac_dim"]

    # 构造算法模型
    model = algo_factory(
        algo_name=algo_name,
        config=config,
        obs_key_shapes=obs_key_shapes,
        ac_dim=ac_dim,
        device=device
    )
    # 加权重
    model.deserialize(ckpt["model"])

    # 归一化统计
    obs_norm_stats = ckpt.get("obs_normalization_stats", None)
    act_norm_stats = ckpt.get("action_normalization_stats", None)

    # RolloutPolicy 包装（做前后处理）
    policy = RolloutPolicy(
        model,
        obs_normalization_stats=obs_norm_stats,
        action_normalization_stats=act_norm_stats,
    )

    # 记录训练时的 obs 键集合，便于我们拼 ob_dict
    obs_keys = list(obs_key_shapes.keys())
    image_obs_keys = [k for k, v in obs_key_shapes.items() if len(v) == 3]  # 粗略判断: 形如 [C,H,W]
    return policy, config, obs_keys, image_obs_keys, shape_meta


def bgr_to_chw_float01(img_bgr, expected_hw=None):
    """把相机帧变成 policy 需要的格式（通常CHW & float[0,1]），并按需要resize。"""
    img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    if expected_hw is not None:
        H, W = expected_hw
        img = cv2.resize(img, (W, H), interpolation=cv2.INTER_AREA)
    img = img.astype(np.float32) / 255.0
    img = np.transpose(img, (2, 0, 1))  # HWC->CHW
    return img


def build_observation(obs_keys, shape_meta, image_obs_keys):
    """
    根据训练时的 obs_keys，从真机传感器拼出 ob_dict（np.ndarray）。
    你需要把不同来源的数据按键名放对。
    """
    ob = {}

    # 1) 图像观测（示例假设键名是 "rgb" 或类似名字；请用你的实际键）
    for k in image_obs_keys:
        H, W = shape_meta["all_shapes"][k][1], shape_meta["all_shapes"][k][2]  # [C,H,W]
        frame_bgr = get_rgb_frame()  # 你的相机函数
        ob[k] = bgr_to_chw_float01(frame_bgr, expected_hw=(H, W))

    # 2) 本体/末端/夹爪（下方举三种常见键名，请根据你的 ckpt 实际键名替换）
    #   - robomimic 常见： "robot0_eef_pos", "robot0_eef_quat", "robot0_gripper_qpos"
    #   - 或者把它们拼成一个向量： "robot0_proprio-state"
    if "robot0_eef_pos" in obs_keys:
        ob["robot0_eef_pos"] = get_eef_pose()[:3].astype(np.float32)
    if "robot0_eef_quat" in obs_keys:
        ob["robot0_eef_quat"] = get_eef_pose()[3:].astype(np.float32)
    if "robot0_gripper_qpos" in obs_keys:
        ob["robot0_gripper_qpos"] = np.array([get_gripper_state()], dtype=np.float32)

    if "robot0_joint_pos" in obs_keys:
        ob["robot0_joint_pos"] = get_joint_positions().astype(np.float32)

    if "robot0_proprio-state" in obs_keys:
        # 例：自己拼一个 [eef_pos(3), eef_quat(4), gripper(1)]
        eef = get_eef_pose()
        grip = np.array([get_gripper_state()], dtype=np.float32)
        ob["robot0_proprio-state"] = np.concatenate([eef[:3], eef[3:], grip], axis=0).astype(np.float32)

    # 3) 其他键（如目标嵌入、语言、深度等）请按 shape_meta/all_obs_keys 对应补齐
    #    如果某键训练时存在，这里必须提供（哪怕是占位零向量，也要符合维度）

    # 最终确保所有键都填了
    for k in obs_keys:
        if k not in ob:
            # 用全零占位（不建议长期这么做，只为快速打通管线）
            shape = shape_meta["all_shapes"][k]
            ob[k] = np.zeros(shape, dtype=np.float32)
    return ob


def map_action_to_robot(action, mode="cartesian_delta", clip=None):
    """
    把 policy 的动作向量映射到真机控制命令。
    具体维度和含义必须与训练环境一致！下面给两个常见例子：
    """
    if clip is not None:
        low, high = clip
        action = np.clip(action, low, high)

    if mode == "cartesian_delta":
        # 例：动作是 [dx, dy, dz, d_rx, d_ry, d_rz, gripper]
        delta_pose = action[:6]  # 平移+旋转增量（旋转可用小角度向量/RPY）
        grip_cmd = action[6] if action.shape[0] > 6 else 0.0
        return dict(type="cartesian_delta", delta=delta_pose, gripper=grip_cmd)

    elif mode == "joint_velocity":
        # 例：前 n 维是关节速度，最后一维是夹爪
        # 请按你的机器人关节数替换 n_j
        n_j = 7
        qd = action[:n_j]
        grip_cmd = action[n_j] if action.shape[0] > n_j else 0.0
        return dict(type="joint_velocity", qd=qd, gripper=grip_cmd)

    else:
        raise ValueError(f"Unknown control mode: {mode}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, required=True)
    parser.add_argument("--fps", type=float, default=10.0, help="控制循环频率")
    parser.add_argument("--cuda", action="store_true", help="用GPU推理")
    parser.add_argument("--control_mode", type=str, default="cartesian_delta",
                        choices=["cartesian_delta", "joint_velocity"])
    args = parser.parse_args()

    device = TorchUtils.get_torch_device(try_to_use_cuda=args.cuda)

    policy, config, obs_keys, image_obs_keys, shape_meta = load_from_ckpt(args.ckpt, device)
    policy.start_episode()  # 初始化策略的内部状态（RNN等）

    print("== Obs keys expected by policy ==")
    print(obs_keys)

    dt = 1.0 / args.fps
    try:
        while robot_is_ok():
            t0 = time.time()

            # 1) 拼装观测字典（np.ndarray）
            ob_dict = build_observation(obs_keys, shape_meta, image_obs_keys)

            # 2) 可选：目标（若训练是 goal-conditioned）。否则传 None。
            goal_dict = None  # 或者按照训练数据构造

            # 3) 策略前向 → 动作（np.ndarray）
            with torch.no_grad():
                action = policy(ob=ob_dict, goal=goal_dict)  # 自动做(反)归一化

            # 4) 映射并下发到真机
            cmd = map_action_to_robot(action, mode=args.control_mode,
                                      clip=(-1.0*np.ones_like(action), 1.0*np.ones_like(action)))

            if cmd["type"] == "cartesian_delta":
                send_cartesian_delta_command(cmd["delta"], dt=dt)
                send_gripper_command(cmd["gripper"])
            elif cmd["type"] == "joint_velocity":
                send_joint_velocity_command(cmd["qd"])
                send_gripper_command(cmd["gripper"])

            # 5) 控制频率
            sleep_t = dt - (time.time() - t0)
            if sleep_t > 0:
                time.sleep(sleep_t)

    except KeyboardInterrupt:
        print("Stopped by user.")
    finally:
        # 这里做安全停机 / 复位
        send_joint_velocity_command(np.zeros(7))  # 示例
        send_gripper_command(0.0)


if __name__ == "__main__":
    main()
