# -*- coding: UTF-8 -*-

import argparse
import json
import numpy as np
import time

import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as Data

from data.data_loaders import data_loader
from logger.logger import get_logger
from model.rhi_ik import HierarchicalInverseModel, ResidualHierarchicalInverseModel
from model.init_model import init_parameters_hierarchicalunit

from utils.error_calculate import batch_rotation_error_deg_torch
import matplotlib.pyplot as plt
from utils.plot import plot_loss_curve_cn
from robot.robot import Robot

def is_normalized(array):
    return np.all((array >= 0.0) & (array <= 1.0))

def save_experiment_results(args, train_losses, position_losses, orientation_losses, model, path):
 
    results = {
        "train_losses": train_losses,
        "posiiton_losses": position_losses,
        "orientation_losses": orientation_losses,
        "parameters": vars(args)
    }
    with open(path + "_results.json", "w") as f:
        json.dump(results, f, indent=4)

    torch.save(model.state_dict(), path + "_model.pth")
    plot_loss_curve_cn(position_losses, orientation_losses, save_path=path + "_loss_curve.png")

def iterative_sampling_and_training(args, train_loader, test_loader, model, robot:Robot,
                                    optimizer, device, logger):

    train_losses = []  
    position_losses = []   
    orientation_losses = []
    min_position_loss = float('inf')
    min_orientation_loss = float('inf')

    for iteration in range(args.max_iteration):
        trainset_input = []
        trainset_output = []
        
        # sampling
        print('-----------------------')
        print('Iteration {}: start sampling...'.format(iteration + 1))
        
        start = time.time()        
        # inverse inference
        model.eval()
        with torch.no_grad():
            for joints, goal in train_loader:
                goal = goal.to(device)
                output = model(goal)
                trainset_output.append(output)  
        
        # parallel: forward simulation
        trainset_output = torch.cat(trainset_output, dim=0)
        joints_cp = robot.joints_cur_denormalize_tensor(trainset_output.clone())
        joints_rad = torch.deg2rad(joints_cp) 
        xyz, rpy = robot.forward_kinematics_torch(joints_rad)
        xyzrpy = torch.cat((xyz, rpy), dim=1)
        xyzrpy = robot.xyzrpy_normalize_tensor(xyzrpy)
        trainset_input = xyzrpy
        
        print('sampling time: {:.2f}'.format(time.time() - start))
        print('Iteration {}: sampling end'.format(iteration + 1))
        print('-----------------------')

        # update trainset
        dataset_sample = Data.TensorDataset(torch.Tensor(trainset_input), torch.Tensor(trainset_output))
        sample_train_loader = Data.DataLoader(dataset_sample, batch_size=args.batch_size, shuffle=True)

        # training
        model.train()
        total_loss = 0
        for epoch in range(1, args.epochs + 1):
            for batch_idx, (data, target) in enumerate(sample_train_loader):
                data, target = data.to(device), target.to(device)
                
                output = model(data)
                loss_joints = nn.MSELoss()(output, target)
                # weights = torch.tensor([1.5, 1.0, 0.8, 0.8, 1.0, 1.5], device=output.device)
                # loss_joints = weighted_mse_loss(output, target, weights)
                                
                loss = loss_joints
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                #loss = nn.MSELoss()(output, target)
                total_loss += loss.item()
                if batch_idx % args.log_interval == 0:
                    print('Iteration: {} \tTrain Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.8f}'.format(
                        iteration + 1, epoch, batch_idx * len(data), len(sample_train_loader.dataset),
                        100. *batch_idx / len(sample_train_loader), loss))

        avg_train_loss = total_loss / len(sample_train_loader)
        train_losses.append(avg_train_loss)  
        # testing
        position_loss , orientation_loss =  test_gpu(model, test_loader, robot, -1, logger, device)
        if position_loss< min_position_loss and orientation_loss < min_orientation_loss:
            min_position_loss = position_loss
            min_orientation_loss = orientation_loss
            torch.save(model.state_dict(), args.model_path + "_model_best.pth")
            print("model_Saved")
        
        position_losses.append(position_loss)
        orientation_losses.append(orientation_loss)
        print('Iteration: {} \tTrain Loss: {:.8f}'.format(iteration + 1, avg_train_loss))
        print('robot:', robot.robot_id)
        
    # if args.save_model:
    #     # torch.save(model, './saved/models/inverse_model_iterative_sampling.pkl')
    #     torch.save(model, args.model_path)
    save_experiment_results(args, train_losses, position_losses, orientation_losses, model, args.model_path)

def test_gpu(model, test_loader, robot:Robot, iteration, logger, device, is_testset=True):
    model.eval()

    all_goal = []
    all_pred_joints = []

    with torch.no_grad():
        for joints, goal in test_loader:
            goal = goal.to(device, non_blocking=True)
            pred = model(goal)                      
            all_goal.append(goal)
            all_pred_joints.append(pred)

    goals = torch.cat(all_goal, dim=0)              # (N, 6) or (N, xyzrpy_dim)
    pred_joints_norm = torch.cat(all_pred_joints, dim=0)  # (N, DoF)

    pred_joints_denorm = robot.joints_cur_denormalize_tensor(pred_joints_norm) 
    pred_joints_rad = torch.deg2rad(pred_joints_denorm)

    xyz_pred, rpy_pred = robot.forward_kinematics_torch(pred_joints_rad)  

    goals_denorm = robot.xyzrpy_denormalize_tensor(goals)  
    xyz_goal = goals_denorm[:, :3]
    rpy_goal = goals_denorm[:, 3:]                   


    pos_err = torch.linalg.norm(xyz_goal - xyz_pred, dim=1).mean()  

    ori_err_deg = batch_rotation_error_deg_torch(rpy_goal, rpy_pred).mean()        


    if is_testset:
        logger.info('Iteration: {} \tTest set Position Error (cm): {:.4f} \tOrientation Error (degree): {:.4f}'.format(
            iteration + 1, pos_err.item() * 100.0, ori_err_deg.item()
        ))
    else:
        logger.info('Iteration: {} \tTrain set Position Error (cm): {:.4f} \tOrientation Error (degree): {:.4f}'.format(
            iteration + 1, pos_err.item() * 100.0, ori_err_deg.item()
        ))

    return pos_err.item() * 100.0, ori_err_deg.item()


def parse_arg():
    parser = argparse.ArgumentParser(description='parse argments')
    parser.add_argument('--robot', type=str, default='piper', 
                        help='robot type: ur3 / panda / piper')
    parser.add_argument('--seed', type=int, default=0,
                        help='random seed')
    parser.add_argument('--max-iteration', type=int, default=400,
                        help='max iteration')                    
    parser.add_argument('--epochs', type=int, default=10,
                        help='number of epochs to train')
    parser.add_argument('--batch-size', type=int, default=128*500,
                        help='input batch size for training')
    parser.add_argument('--test-batch-size', type=int, default=512*5,
                        help='input batch size for testing and inferencing')
    parser.add_argument('--lr', type=float, default=0.0015,
                        help='learning rate')
    parser.add_argument('--log-interval', type=int, default=100,
                        help='how many batches to wait before logging training status')
    parser.add_argument('--num-workers', type=int, default=1,
                        help='data loader num workers')
    parser.add_argument('--data-drop-rate', type=float, default=0.5,
                        help='data drop rate')
    parser.add_argument('--save-model', action='store_true', default=True,
                        help='saving the current Model')
    parser.add_argument('--data-path', type=str, default='',
                        help='data path')
    parser.add_argument('--model-path', type=str, default='',
                        help='model path')
    parser.add_argument('--log-path', type=str, default='',
                        help='log path')
    parser.add_argument('--net-arch', nargs='+', type=int,
                        help='network architecture for the model')
    parser.add_argument('--conv-coefficient', type=float, default=0.07,
                        help='draw random samples from a multivariate normal distribution')
    parser.add_argument('--processes', type=int, default=1,
                        help='number of processes for data sampling') 
    parser.add_argument('--use-hierarchy', action='store_true', default=True,
                        help='enable hierarchical training')
    parser.add_argument('--save-model-interval', type=int, default=40,
                        help='save-model-interval')
    parser.add_argument('--continue-training', action='store_true', default=False,
                        help='enable continue training')
    args = parser.parse_args()
    return args


def main():
    args = parse_arg()
    robot = Robot(robot_id=args.robot)
    args.model_path = f'./saved/models/{args.robot}/' + 'inverse_model_iterative_sampling'

    if len(args.log_path) == 0:
        time_now = time.strftime("%Y-%m-%d-%H_%M_%S", time.localtime(time.time()))
        args.log_path = './saved/log/' + 'inverse_model_iterative_sampling_' + time_now + '.log'
    logger = get_logger(args.log_path)
    
    # saves arguments to config file
    args_path = './saved/log/' + 'args_inverse_model_iterative_sampling_' + time_now + '.json'
    argparse_dict = vars(args)
    with open(args_path, 'w', encoding='utf-8') as f:
        f.write(json.dumps(argparse_dict, ensure_ascii=False, indent=4))
    
    # set seed
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    if torch.cuda.is_available():
        device = torch.device('cuda')
        torch.cuda.manual_seed(args.seed)
    else:
        device = torch.device('cpu')


    trainset = data_loader('dataset/piper/all_area/generated_data_train.txt', robot=robot)   
    testset = data_loader('dataset/piper/all_area/generated_data_test.txt', robot=robot)


    train_loader = Data.DataLoader(
        dataset = trainset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle = True)
    test_loader = Data.DataLoader(
        dataset = testset,
        batch_size=args.test_batch_size,
        num_workers=args.num_workers,
        shuffle=False)    

    if args.continue_training:
        model = torch.load().to(device)
    else:
        model = ResidualHierarchicalInverseModel(dof=robot.const["DOF"], net_arch=args.net_arch).to(device)
        init_parameters_hierarchicalunit(model)

    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    iterative_sampling_and_training(args, train_loader, test_loader, model, robot, optimizer, device, logger)


if __name__ == "__main__":
    start = time.time()
    main()
    end = time.time()
    h, m, s = (end - start) // 3600, (end - start) % 3600 // 60, (end - start) % 3600 % 60 
    print("Time used: {:.0f} hour, {:.0f} minute, {:.0f} second\n".format(h, m, s))