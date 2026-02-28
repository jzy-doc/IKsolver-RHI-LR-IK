from utils.normalize import *
from robot.piper import get_piper_dh as get_dh, piper_forward_kinematics_torch as forward_kinematics_torch
import torch.nn as nn

def clamp_outputs(outputs: torch.Tensor,
                  min_restricted, max_restricted) -> torch.Tensor:
    min_restricted = torch.as_tensor(min_restricted, device=outputs.device, dtype=outputs.dtype)
    max_restricted = torch.as_tensor(max_restricted, device=outputs.device, dtype=outputs.dtype)

    return torch.clamp(outputs, min=min_restricted, max=max_restricted)

class Model_Combined(nn.Module):
    def __init__(self, rhi_ik_model, lr_ik_model_list, get_dh_fn, 
                 joints_denorm_fn, joints_norm_fn,
                 xyzrpy_denorm_fn, delta_xyzrpy_norm_fn, delta_joints_denorm_fn):
        super().__init__()
        self.rhi_ik = rhi_ik_model
        self.lr_ik1  = lr_ik_model_list[0]
        self.lr_ik2  = lr_ik_model_list[1]
        self.get_dh = get_dh_fn
        self.joints_denorm = joints_denorm_fn
        self.joints_norm   = joints_norm_fn
        self.xyzrpy_denorm = xyzrpy_denorm_fn
        self.delta_xyzrpy_norm = delta_xyzrpy_norm_fn
        self.delta_joints_denorm = delta_joints_denorm_fn

    def forward(self, input_data):
        joints_norm_seed = self.rhi_ik(input_data)            
        joints_deg_seed  = self.joints_denorm(joints_norm_seed) 
        joints_rad_seed  = torch.deg2rad(joints_deg_seed)

        xyz_cur, rpy_cur = forward_kinematics_torch(joints_rad_seed, *self.get_dh())
        xyzrpy_cur       = torch.cat((xyz_cur, rpy_cur), dim=1)

        xyzrpy_target    = self.xyzrpy_denorm(input_data)
        delta_xyzrpy     = xyzrpy_target - xyzrpy_cur
        delta_xyzrpy_n   = self.delta_xyzrpy_norm(delta_xyzrpy)

        lr_inputs        = torch.cat((joints_norm_seed, delta_xyzrpy_n), dim=1)
        delta_q_pred     = self.lr_ik1(lr_inputs)
        delta_q_deg      = self.delta_joints_denorm(delta_q_pred)

        joints_deg_final = joints_deg_seed + delta_q_deg
        # joints_deg_final = clamp_outputs(joints_deg_final, MIN_JOINTS_UESD, MAX_JOINTS_UESD)
        
        joints_rad_final = torch.deg2rad(joints_deg_final)
        xyz_cur, rpy_cur = forward_kinematics_torch(joints_rad_final, *self.get_dh())
        xyzrpy_cur       = torch.cat((xyz_cur, rpy_cur), dim=1)
        delta_xyzrpy     = xyzrpy_target - xyzrpy_cur
        delta_xyzrpy_n   = self.delta_xyzrpy_norm(delta_xyzrpy)
        
        lr_inputs_final = torch.cat((self.joints_norm(joints_deg_final), delta_xyzrpy_n), dim=1)
        delta_q_pred_final = self.lr_ik2(lr_inputs_final)
        delta_q_deg_final  = self.delta_joints_denorm(delta_q_pred_final)
        joints_deg_final = joints_deg_final + delta_q_deg_final
        
        # joints_deg_final = clamp_outputs(joints_deg_final, MIN_JOINTS_UESD, MAX_JOINTS_UESD)
        return joints_deg_final

    