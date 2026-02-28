import torch
import numpy as np
from scipy.spatial.transform import Rotation as R

def euler_to_matrix_np(euler_angles):
    return R.from_euler('xyz', euler_angles, degrees=True).as_matrix()

def rotation_error_deg_np(euler_gt, euler_pred):
    R_gt = euler_to_matrix_np(euler_gt)
    R_pred = euler_to_matrix_np(euler_pred)
    cos_angle = (np.trace(np.dot(R_gt.T, R_pred)) - 1) / 2
    cos_angle = np.clip(cos_angle, -1.0, 1.0)
    angle_rad = np.arccos(cos_angle)
    return np.degrees(angle_rad)


def euler_to_matrix_torch(euler_angles):
    euler_rad = torch.deg2rad(euler_angles)
    roll, pitch, yaw = euler_rad[:, 0], euler_rad[:, 1], euler_rad[:, 2]
    cos_r, sin_r = torch.cos(roll), torch.sin(roll)
    cos_p, sin_p = torch.cos(pitch), torch.sin(pitch)
    cos_y, sin_y = torch.cos(yaw), torch.sin(yaw)
    batch_size = euler_angles.shape[0]

    device = euler_angles.device

    Rx = torch.zeros((batch_size, 3, 3), device=device)
    Ry = torch.zeros((batch_size, 3, 3), device=device)
    Rz = torch.zeros((batch_size, 3, 3), device=device)

    Rx[:, 0, 0] = 1
    Rx[:, 1, 1] = cos_r
    Rx[:, 1, 2] = -sin_r
    Rx[:, 2, 1] = sin_r
    Rx[:, 2, 2] = cos_r

    Ry[:, 0, 0] = cos_p
    Ry[:, 0, 2] = sin_p
    Ry[:, 1, 1] = 1
    Ry[:, 2, 0] = -sin_p
    Ry[:, 2, 2] = cos_p

    Rz[:, 0, 0] = cos_y
    Rz[:, 0, 1] = -sin_y
    Rz[:, 1, 0] = sin_y
    Rz[:, 1, 1] = cos_y
    Rz[:, 2, 2] = 1

    R = torch.bmm(Rz, torch.bmm(Ry, Rx))
    return R

# def batch_rotation_error_deg_torch(euler_gt, euler_pred):
#     R_gt = euler_to_matrix_torch(euler_gt)
#     R_pred = euler_to_matrix_torch(euler_pred)
#     R_diff = torch.bmm(R_gt.transpose(1, 2), R_pred)
#     trace = R_diff[:, 0, 0] + R_diff[:, 1, 1] + R_diff[:, 2, 2]
#     cos_theta = (trace - 1) / 2
#     cos_theta = torch.clamp(cos_theta, -1.0, 1.0)
#     angle_rad = torch.acos(cos_theta)
#     angle_deg = torch.rad2deg(angle_rad)
#     return angle_deg

def _wrap180(x: torch.Tensor) -> torch.Tensor:
    return (x + 180.0) % 360.0 - 180.0

def _pi_flip_equivalent(euler_deg: torch.Tensor) -> torch.Tensor:
    """
    ZYX  (yaw, pitch, roll) ：
    (ψ, θ, φ) ≡ (ψ+180, -θ, φ+180)
    """
    return torch.stack([
        _wrap180(euler_deg[:, 0] + 180.0),   # yaw + 180
        _wrap180(-euler_deg[:, 1]),          # -pitch
        _wrap180(euler_deg[:, 2] + 180.0),   # roll + 180
    ], dim=1)

@torch.no_grad()
def batch_rotation_error_deg_torch(euler_gt: torch.Tensor,
                                   euler_pred: torch.Tensor) -> torch.Tensor:
    candidates = [ _wrap180(euler_pred), _pi_flip_equivalent(euler_pred) ]
    cand_stack = torch.stack(candidates, dim=0)  # [K,B,3], K=2

    R_gt = euler_to_matrix_torch(_wrap180(euler_gt))  # [B,3,3]

    all_errors = []
    for k in range(cand_stack.size(0)):
        R_pred = euler_to_matrix_torch(cand_stack[k])
        R_diff = torch.bmm(R_gt.transpose(1, 2), R_pred)
        trace = R_diff[:, 0, 0] + R_diff[:, 1, 1] + R_diff[:, 2, 2]
        cos_th = torch.clamp((trace - 1.0) / 2.0, -1.0, 1.0)
        ang_deg = torch.rad2deg(torch.acos(cos_th))
        all_errors.append(ang_deg)

    all_errors = torch.stack(all_errors, dim=0)
    return all_errors.min(dim=0)[0]

