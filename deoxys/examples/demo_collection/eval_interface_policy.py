# inference_pushT.py
import json
import torch
import numpy as np
from typing import Dict, Any
from robomimic.utils.file_utils import policy_from_checkpoint
import robomimic.utils.obs_utils as ObsUtils
from robomimic.config import config_factory 
from robomimic.utils.file_utils import policy_from_checkpoint

def load_config(cfg_path: str) -> Dict[str, Any]:
    with open(cfg_path, "r") as f:
        return json.load(f)

def init_obs_utils_from_config(cfg_path: str, algo_name: str = None):
    
    ext_cfg = json.load(open(cfg_path, "r"))

    if algo_name is None:
        algo_name = ext_cfg["algo_name"]

    config = config_factory(algo_name)

    with config.values_unlocked():
        config.update(ext_cfg)

    # 3) 初始化 ObsUtils
    ObsUtils.initialize_obs_utils_with_config(config)

    return config

class RoboMimicPolicyWrapper:
    
    def __init__(self, policy, device: str = "cuda"):
        self.policy = policy
        self.device = device

    @torch.no_grad()
    def __call__(self, obs_dict: Dict[str, Any]) -> np.ndarray:
 
        out = self.policy(obs_dict)

        if isinstance(out, dict):
            if "action" in out:
                act = out["action"]
            elif "actions" in out:
                act = out["actions"]
            else:
                
                act = next((v for v in out.values() if torch.is_tensor(v)), None)
                if act is None:
                    raise RuntimeError("无法在策略输出中找到 action。")
        else:
            act = out
        if act.ndim == 2 and act.shape[0] == 1:
            act = act[0]
        return act

def build_policy(ckpt_path: str, cfg_path: str, device: str = None):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    config = init_obs_utils_from_config(cfg_path)

    
    policy,_ = policy_from_checkpoint(device=device,ckpt_path=ckpt_path)
    policy.policy.reset()
    return RoboMimicPolicyWrapper(policy, device=device)

def make_predict_fn(ckpt_path: str, cfg_path: str, device: str = None):
   
    wrapped = build_policy(ckpt_path, cfg_path, device=device)
    def predict(obs_dict: Dict[str, Any]) -> np.ndarray:
        return wrapped(obs_dict)
    return predict

if __name__ == "__main__":
   
    hdf5_path = "/home/hinton/deoxys_gyh/example_data/pusht.hdf5"
    ckpt_path = "/home/hinton/deoxys_gyh/checkpoints/model_epoch_1000.pth"
    cfg_path = "/home/hinton/deoxys_gyh/robomimic/robomimic/exps/templates/diffusion_policy_pusht.json"

    predict = make_predict_fn(ckpt_path, cfg_path, device="cuda")
    import h5py
    with h5py.File(hdf5_path, "r") as f:
        demo = f["data/demo_0"]
        obs_grp = demo["obs"]
        print("Keys in obs:", list(obs_grp.keys()))
        obs_dict = {}

        for k in obs_grp.keys():
            data = obs_grp[k]
            
            combined = data[0:2] 
            obs_dict[k] = combined
        

    action = predict(obs_dict)
    print("Predicted action:", action, "shape:", action.shape)