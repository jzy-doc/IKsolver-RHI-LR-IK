import sys
sys.path.append("..")

import math
import numpy as np

import torch
import torch.utils.data as Data
from robot.robot import Robot


def read_arm_data(file_path, robot:Robot):
    joint_angles = []  
    end_effector_data = []  

    with open(file_path, 'r') as file:
        for line in file:
            line = line.strip()

            data = list(map(float, line.split(',')))

            joints = data[:robot.const["DOF"]]
            joint_angles.append(joints)

            end_effector = data[robot.const["DOF"]:]
            end_effector_data.append(end_effector)

    joint_angles = np.array(joint_angles)
    end_effector_data = np.array(end_effector_data)
    # joint_angles_max = np.max(joint_angles, axis=0)  
    # joint_angles_min = np.min(joint_angles, axis=0)
    # end_effector_data_max = np.max(end_effector_data, axis=0)
    # end_effector_data_min = np.min(end_effector_data, axis=0)
    # print(joint_angles_max, joint_angles_min, end_effector_data_max, end_effector_data_min)
    return joint_angles, end_effector_data

def read_arm_data2(file_path, robot:Robot):
    joint_angles = []  
    end_effector_data = []
    delta_q_data = []  

    with open(file_path, 'r') as file:
        for line in file:
            line = line.strip()

            data = list(map(float, line.split(',')))

            joints = data[:robot.const["DOF"]]
            joint_angles.append(joints)
            
            delta_q = data[robot.const["DOF"]:(2*robot.const["DOF"])]
            delta_q_data.append(delta_q)

            end_effector = data[(2*robot.const["DOF"]):]
            end_effector_data.append(end_effector)

    joint_angles = np.array(joint_angles)
    end_effector_data = np.array(end_effector_data)
    delta_q_data = np.array(delta_q_data)
    # joint_angles_max = np.max(joint_angles, axis=0)  
    # joint_angles_min = np.min(joint_angles, axis=0)
    # end_effector_data_max = np.max(end_effector_data, axis=0)
    # end_effector_data_min = np.min(end_effector_data, axis=0)
    # delta_q_data_max = np.max(delta_q_data, axis=0)
    # delta_q_data_min = np.min(delta_q_data, axis=0)
    # print(joint_angles_max, joint_angles_min, end_effector_data_max, end_effector_data_min)
    # print(delta_q_data_max, delta_q_data_min)
    return joint_angles, end_effector_data, delta_q_data


def data_loader(filename, robot:Robot):
    """For iterative sampling
    Returns:
        goalset: Data.TensorDataset
    """
    joints, end_pose = read_arm_data(file_path=filename, robot=robot)
    joints = torch.from_numpy(joints).float()  # numpy to tensor
    end_pose = torch.from_numpy(end_pose).float()  # numpy to tensor
    
    joints_norm, end_pose_norm = robot.joints_cur_normalize_tensor(joints), robot.xyzrpy_normalize_tensor(end_pose)
    goalset = Data.TensorDataset(joints_norm, end_pose_norm)
    return goalset

def data_loader_dymamic(filename, robot:Robot):
    joints, end_pose, delta_q = read_arm_data2(file_path=filename, robot=robot)
    joints = torch.from_numpy(joints).float()  # numpy to tensor
    end_pose = torch.from_numpy(end_pose).float()  # numpy to tensor
    delta_q = torch.from_numpy(delta_q).float()
    
    joints_norm, end_pose_norm, delta_q_norm = robot.joints_cur_normalize_tensor(joints), robot.delta_xyzrpy_normalize_tensor(end_pose), robot.delta_joints_normalize_tensor(delta_q)

    goalset = Data.TensorDataset(joints_norm, end_pose_norm, delta_q_norm)
    return goalset