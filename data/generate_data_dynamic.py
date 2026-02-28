import os
import torch
import numpy as np
from robot.robot import Robot

@torch.no_grad()
def normalize_rpy_deg_torch(rpy_deg: torch.Tensor) -> torch.Tensor:
    return torch.remainder(rpy_deg + 180.0, 360.0) - 180.0

@torch.no_grad()
def fk_xyzrpy_deg(robot: Robot, joints_rad: torch.Tensor) -> torch.Tensor:
    # joints_rad: (N, DOF) on device
    xyz, rpy_deg = robot.forward_kinematics_torch(joints_rad)  # xyz: m, rpy: rad
    rpy_deg = normalize_rpy_deg_torch(rpy_deg)
    return torch.cat((xyz, rpy_deg), dim=1)  # (N, 6)

@torch.no_grad()
def generate_delta_dataset_torch(robot: Robot,
                                 num_base_q: int,
                                 num_delta_per_q: int,
                                 device: torch.device,
                                 base_chunk: int = 1024) -> np.ndarray:

    C = robot.const
    dof = int(C["DOF"])
    min_deg = torch.tensor(C["MIN_JOINTS_UESD"], dtype=torch.float32, device=device)
    max_deg = torch.tensor(C["MAX_JOINTS_UESD"], dtype=torch.float32, device=device)
    dq_range = float(C["DELTA_ANGLE_RANGE"])

    rows_out = []
    for start in range(0, num_base_q, base_chunk):
        B = min(base_chunk, num_base_q - start)

        q_deg = min_deg + (max_deg - min_deg) * torch.rand(B, dof, device=device)
        q_rep_deg = q_deg.repeat_interleave(num_delta_per_q, dim=0)
        delta_deg = (torch.rand(B * num_delta_per_q, dof, device=device) * (2 * dq_range) - dq_range)

        q_rad = torch.deg2rad(q_rep_deg)
        dq_rad = torch.deg2rad(delta_deg)

        xyzrpy_cur = fk_xyzrpy_deg(robot, q_rad)            
        xyzrpy_now = fk_xyzrpy_deg(robot, q_rad + dq_rad)     

        delta_xyzrpy = xyzrpy_now - xyzrpy_cur           
        delta_xyz = delta_xyzrpy[:, :3]
        delta_rpy = normalize_rpy_deg_torch(delta_xyzrpy[:, 3:])            

        row = torch.cat((q_rep_deg, delta_deg, delta_xyz, delta_rpy), dim=1) 
        rows_out.append(row.detach().cpu())

    data = torch.cat(rows_out, dim=0).numpy()  # (num_base_q*num_delta_per_q, 2*DOF+6)
    return data

def save_numpy_csv(data: np.ndarray, output_file: str):
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    np.savetxt(output_file, data, delimiter=",", fmt="%.6f")

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    robot = Robot("piper")

    train_joints_num_samples = 10000
    test_joints_num_samples = 300        
    delta_num_samples = 100           
    base_chunk = 1024                 

    train_data = generate_delta_dataset_torch(robot,
                                        num_base_q=train_joints_num_samples,
                                        num_delta_per_q=delta_num_samples,
                                        device=device,
                                        base_chunk=base_chunk)
    test_data = generate_delta_dataset_torch(robot,
                                        num_base_q=test_joints_num_samples,
                                        num_delta_per_q=delta_num_samples,
                                        device=device,
                                        base_chunk=base_chunk)

    train_output_file = (
        f"./dataset/{robot.robot_id}/"
        f"range_{int(robot.const['DELTA_ANGLE_RANGE'])}_all_area/generated_data_train.txt"
    )
    test_output_file = (
        f"./dataset/{robot.robot_id}/"
        f"range_{int(robot.const['DELTA_ANGLE_RANGE'])}_all_area/generated_data_test.txt"
    )
    
    save_numpy_csv(train_data, train_output_file)
    save_numpy_csv(test_data, test_output_file)
    print(f"save {train_data.shape[0]} , {train_data.shape[1]} to {train_output_file}")
    print(f"save {test_data.shape[0]} , {test_data.shape[1]} to {test_output_file}")
