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
    # rpy_deg = normalize_rpy_deg_torch(rpy_deg)
    return torch.cat((xyz, rpy_deg), dim=1)  # (N, 6)

@torch.no_grad()
def generate_dataset_torch(robot: Robot, num_samples: int, device: torch.device,
                           chunk_size: int = 100000) -> np.ndarray:
    c = robot.const
    dof = int(c["DOF"])

    # 关节上下限（度）
    min_deg = torch.tensor(c["MIN_JOINTS_UESD"], dtype=torch.float32, device=device)
    max_deg = torch.tensor(c["MAX_JOINTS_UESD"], dtype=torch.float32, device=device)

    out_chunks = []

    for start in range(0, num_samples, chunk_size):
        n = min(chunk_size, num_samples - start)
        joints_deg = min_deg + (max_deg - min_deg) * torch.rand(n, dof, device=device)
        joints_rad = torch.deg2rad(joints_deg)

        xyzrpy_deg = fk_xyzrpy_deg(robot, joints_rad)  # (n, 6)
        rows = torch.cat((joints_deg, xyzrpy_deg), dim=1).detach().cpu()  # (n, dof+6)
        out_chunks.append(rows)

    data = torch.cat(out_chunks, dim=0).numpy()
    return data

def save_numpy_csv(data: np.ndarray, output_file: str):
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    np.savetxt(output_file, data, delimiter=",", fmt="%.6f")

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    robot = Robot("piper")

    train_num = 500000
    test_num = 30000

    data_train = generate_dataset_torch(robot, train_num, device, chunk_size=131072)
    data_test = generate_dataset_torch(robot, test_num, device, chunk_size=131072)

    train_output_file = (
        f"./dataset/{robot.robot_id}/"
        f"all_area/generated_data_train.txt"
    )
    test_output_file = (
        f"./dataset/{robot.robot_id}/"
        f"all_area/generated_data_test.txt"
    )

    save_numpy_csv(data_train, train_output_file)
    save_numpy_csv(data_test, test_output_file)
    print(f"save {data_train.shape[0]} to {train_output_file}")
    print(f"save {data_test.shape[0]} to {test_output_file}")
