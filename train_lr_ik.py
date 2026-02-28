# -*- coding: UTF-8 -*-
import argparse
import json
import numpy as np
import time
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as Data
from data.data_loaders import data_loader_dymamic
from model.lr_ik import DynamicModel
from model.init_model import init_parameters_uniform
from utils.error_calculate import batch_rotation_error_deg_torch
from utils.plot import plot_loss_curve_cn
from robot.robot import Robot

def save_experiment_results(args, train_losses, position_losses, rotation_losses, model, path):
    results = {
        "position_losses": position_losses,
        "rotation_losses": rotation_losses,
        "parameters": vars(args)
    }
    with open(path + "_results.json", "w") as f:
        json.dump(results, f, indent=4)

    torch.save(model.state_dict(), path + "_model.pth")
    plot_loss_curve_cn(position_losses, rotation_losses, save_path=path + "_loss_curve.png")

def forward_dh_calculate(robot:Robot, joint_cur, delta_q):
    joint_cur_cp = joint_cur.clone()
    delta_q_cp = delta_q.clone()
    joint_cur_denormalize = robot.joints_cur_denormalize_tensor(joint_cur_cp)
    delta_q_denormalize = robot.delta_joints_denormalize_tensor(delta_q_cp)
    
    # max_per_column, _ = joint_cur_denormalize.max(dim=0)
    # min_per_column, _ = joint_cur_denormalize.min(dim=0)
    # print('joint_cur per-column stats:')
    # for i in range(joint_cur_denormalize.shape[1]):
    #     print(f"  Column {i}: Max = {max_per_column[i].item():.4f}, Min = {min_per_column[i].item():.4f}")
    
    # max_per_column, _ = delta_q_denormalize.max(dim=0)
    # min_per_column, _ = delta_q_denormalize.min(dim=0)
    # print('delta_q per-column stats:')
    # for i in range(delta_q_denormalize.shape[1]):
    #     print(f"  Column {i}: Max = {max_per_column[i].item():.4f}, Min = {min_per_column[i].item():.4f}")
    
    joint_cur_rad = torch.deg2rad(joint_cur_denormalize)
    delta_q_rad = torch.deg2rad(delta_q_denormalize)
    xyz_cur, rpy_cur = robot.forward_kinematics_torch(joint_cur_rad)
    xyz_now, rpy_now = robot.forward_kinematics_torch(joint_cur_rad + delta_q_rad)
    delta_xyz = xyz_now - xyz_cur
    delta_rpy = rpy_now - rpy_cur
    delta_rpy = (delta_rpy + 180)% 360 - 180  
    delta_xyzrpy = torch.cat((delta_xyz, delta_rpy), dim=1)  # [batch_size, 6]
    xyzrpy_now = torch.cat((xyz_now, rpy_now), dim=1)  # [batch_size, 6] 
    delta_xyzrpy = robot.delta_xyzrpy_normalize_tensor(delta_xyzrpy)  
    return delta_xyzrpy, xyzrpy_now  

def iterative_sampling_and_training(args, model, train_loader, test_loader, robot:Robot, optimizer, device):
    model_params_to_update = [param for param in model.parameters() if param.requires_grad]
    optimizer = optim.Adam(model_params_to_update, lr=args.lr)  
    
    train_losses = []  
    position_losses = []  
    rotation_losses = [] 
    init_cov_coefficient = 0.07
    cov_coefficient = init_cov_coefficient

    for iteration in range(args.max_iteration):
        delta_q_set = []
        cur_q_set = []
        
        # sampling
        print('-----------------------')
        print('Iteration {}: start sampling...'.format(iteration + 1))
        print('delta_angle_range', robot.const["DELTA_ANGLE_RANGE"])
        start = time.time()
        
        # inverse inference
        model.eval()
        with torch.no_grad():
            for joints, xyzrpy, delta_q in train_loader:
                xyzrpy = xyzrpy.to(device)
                joints = joints.to(device)
                model_input = torch.cat((joints, xyzrpy), dim=1)
                output = model(model_input)
                cur_q_set += joints.cpu().numpy().tolist()
                delta_q_set += output.cpu().numpy().tolist()
        
        joints_cur = torch.tensor(cur_q_set)
        delta_q_output = torch.tensor(delta_q_set)
        
        # guass sampling
        cov_matrix = np.diag([cov_coefficient] * robot.const["DOF"])
        delta_q_output = np.array([np.random.multivariate_normal(delta_q_row, cov_matrix) for delta_q_row in delta_q_output.numpy()])
        delta_q_output = torch.tensor(delta_q_output, dtype=torch.float32)
        delta_q_output = torch.clamp(delta_q_output, 0, 1)
        cov_coefficient = cov_coefficient - init_cov_coefficient*0.002
        
        # delta_q_output_cp = delta_q_output.clone()
        # delta_q_output_cp = sampling_delta_q_normalize(delta_q_output_cp, min_values, max_values)
        joints_cur = joints_cur.to(device)
        delta_q_output = delta_q_output.to(device)
        
        delta_xyzrpy, _ = forward_dh_calculate(robot, joints_cur, delta_q_output)
        
        print('Iteration {}: sampling end'.format(iteration + 1))
        print('sampling time: {:.2f}'.format(time.time() - start))
        print('-----------------------')

        # update trainset
        dataset_sample = Data.TensorDataset(torch.Tensor(joints_cur), torch.Tensor(delta_xyzrpy), torch.Tensor(delta_q_output))
        sample_train_loader = Data.DataLoader(dataset_sample, batch_size=args.batch_size, shuffle=True)

        # training
        model.train()
        total_loss = 0  
        for epoch in range(1, args.epochs + 1):      
            for batch_idx, (data_joints, data_delta_xyzrpy, target) in enumerate(sample_train_loader):
                data_joints, data_delta_xyzrpy, target = data_joints.to(device), data_delta_xyzrpy.to(device), target.to(device)
                
                data = torch.cat((data_joints, data_delta_xyzrpy), dim=1)
                optimizer.zero_grad()
                output = model(data)
                joint_loss = nn.MSELoss()(output, target)

                loss = joint_loss 
                loss.backward(retain_graph=True)
                optimizer.step()

                total_loss += loss.item()
                

            print('Epoch: {} \tLoss: {:.8f} '.format(
                epoch, loss.item()))

        avg_loss = total_loss / (len(sample_train_loader) * int(args.epochs))
        train_losses.append(avg_loss)  
        
        # testing
        print('-----------------------')
        print('iteration time: {:.2f}'.format(time.time() - start))
        total_count, correct_count, max_error, min_error, total_error = test(model, test_loader, robot, device)
        
        mean_error_xyz = total_error[0]/ total_count *100 
        mean_error_rpy = total_error[1]/ total_count
        mean_error_delta_q = total_error[2] / total_count

        print("robot", robot.robot_id)
        print(f'total_test_num: {total_count}')
        print(f'correct_num(error<0.01): {correct_count}')
        print(f'max_error: {max_error[0]:.4f} {max_error[1]:.4f} {max_error[2]:.4f}')
        print(f'min_error: {min_error[0]:.6f} {min_error[1]:.6f} {min_error[2]:.4f}')
        print(f'total_error:{total_error[0]:.4f} {total_error[1]:.4f} {total_error[2]:.4f}')
        print(f'avg_error:"{mean_error_xyz:.4f} {mean_error_rpy:.4f} {mean_error_delta_q:.4f}')
       
        position_losses.append(mean_error_xyz)
        rotation_losses.append(mean_error_rpy) 
        
        # torch.save(model, './saved/models/inverse_model_iterative_sampling.pkl')
        torch.save(model, args.model_path + "_model.pth")

    save_experiment_results(args, train_losses, position_losses, rotation_losses, model, args.model_path)


def test(model, test_loader, robot:Robot, device, is_testset=True):
    model.eval()

    total_error_xyz = torch.tensor(0.0, device=device)
    total_error_rpy = torch.tensor(0.0, device=device)
    total_error_delta_q = torch.tensor(0.0, device=device)
    
    max_error_xyz = torch.tensor(0.0, device=device)
    min_error_xyz = torch.tensor(float('inf'), device=device)
    max_error_rpy = torch.tensor(0.0, device=device)
    min_error_rpy = torch.tensor(float('inf'), device=device)
    max_error_delta_q = torch.tensor(0.0, device=device)
    min_error_delta_q = torch.tensor(float('inf'), device=device)

    correct_count = torch.tensor(0, device=device)
    total_count = torch.tensor(0, device=device)

    with torch.no_grad():
        all_joints = []
        all_delta_xyzrpy_gt = []
        all_delta_q = []
        
        for joints, delta_xyzrpy_gt, delta_q in test_loader:
            all_joints.append(joints.to(device))
            all_delta_xyzrpy_gt.append(delta_xyzrpy_gt.to(device))
            all_delta_q.append(delta_q.to(device))
        
        joints = torch.cat(all_joints, dim=0)
        delta_xyzrpy_gt = torch.cat(all_delta_xyzrpy_gt, dim=0)
        delta_q_gt = torch.cat(all_delta_q, dim=0)

        input_data = torch.cat((joints, delta_xyzrpy_gt), dim=1)
        output = model(input_data)  
        
        delta_q_error = robot.const["DELTA_ANGLE_RANGE"] * torch.sqrt(torch.sum((delta_q_gt - output) ** 2, dim=1))

        delta_xyzrpy, xyzrpy_now = forward_dh_calculate(robot, joints, output)

        delta_xyzrpy_gt = robot.delta_xyzrpy_denormalize_tensor(delta_xyzrpy_gt)
        delta_xyzrpy = robot.delta_xyzrpy_denormalize_tensor(delta_xyzrpy)

        error_position = torch.sqrt(torch.sum((delta_xyzrpy_gt[:, :3] - delta_xyzrpy[:, :3]) ** 2, dim=1))
        error_rpy = batch_rotation_error_deg_torch(delta_xyzrpy_gt[:, 3:]+xyzrpy_now[:, 3:], delta_xyzrpy[:, 3:]+xyzrpy_now[:, 3:])

        correct_mask = (error_position < 0.01) & (error_rpy < 10)
        correct_count += correct_mask.sum()
        total_count += error_position.size(0)

        max_error_xyz = torch.max(max_error_xyz, error_position.max())
        min_error_xyz = torch.min(min_error_xyz, error_position.min())
        max_error_rpy = torch.max(max_error_rpy, error_rpy.max())
        min_error_rpy = torch.min(min_error_rpy, error_rpy.min())
        max_error_delta_q = torch.max(max_error_delta_q, delta_q_error.max())
        min_error_delta_q = torch.min(min_error_delta_q, delta_q_error.min())

        total_error_xyz += error_position.sum()
        total_error_rpy += error_rpy.sum()
        total_error_delta_q += delta_q_error.sum()

    max_error = [max_error_xyz.item(), max_error_rpy.item(), max_error_delta_q.item()]
    min_error = [min_error_xyz.item(), min_error_rpy.item(), min_error_delta_q.item()]
    total_error = [total_error_xyz.item(), total_error_rpy.item(), total_error_delta_q.item()]
    
    return total_count.item(), correct_count.item(), max_error, min_error, total_error 

def parse_arg():
    parser = argparse.ArgumentParser(description='parse argments')
    parser.add_argument('--robot', type=str, default='piper', 
                        help='robot type: ur3 / panda / piper')
    parser.add_argument('--seed', type=int, default=10,
                        help='random seed')
    parser.add_argument('--max-iteration', type=int, default=500,
                        help='max iteration')                    
    parser.add_argument('--epochs', type=int, default=10,
                        help='number of epochs to train')
    parser.add_argument('--batch-size', type=int, default=4096*4,
                        help='input batch size for training')
    parser.add_argument('--test-batch-size', type=int, default=1,
                        help='input batch size for testing and inferencing')
    # parser.add_argument('--lr', type=float, default=0.0001,
    #                     help='learning rate')
    # parser.add_argument('--lr', type=float, default=0.0005,
    #                     help='learning rate')
    parser.add_argument('--lr', type=float, default=0.001,
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
    parser.add_argument('--use-hierarchy', action='store_true', default=False,
                        help='enable hierarchical training')
    parser.add_argument('--save-model-interval', type=int, default=40,
                        help='save-model-interval')
    parser.add_argument('--continue-training', action='store_true', default=False,
                        help='enable continue training')
    args = parser.parse_args()
    return args

def main():
    args = parse_arg()
    use_gpu = True
    robot = Robot(args.robot)
    
    # set seed
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    if torch.cuda.is_available() and use_gpu:
        device = torch.device('cuda')
        torch.cuda.manual_seed(args.seed)
    else:
        device = torch.device('cpu')

    if args.continue_training:
        model = torch.load(args.model_path).to(device)
    else:
        model = DynamicModel(input_dim=(robot.const["DOF"]+6), joint_output_dim=robot.const["DOF"]).to(device)
        print('model:', model)
        print('model parameters:', sum(p.numel() for p in model.parameters()))
        
        init_parameters_uniform(model)
        # init_parameters_uniform(model, bias_fill_zeros=True)
        # init_parameters_normal(model)

    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    
    trainset = data_loader_dymamic('dataset/piper/range_5_all_area/generated_data_train.txt', robot=robot)   
    testset = data_loader_dymamic('dataset/piper/range_5_all_area/generated_data_test.txt', robot=robot)
        
    train_loader = Data.DataLoader(
        trainset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle = True)
    test_loader = Data.DataLoader(
        testset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=False)
    iterative_sampling_and_training(args, model, train_loader, test_loader, robot, optimizer, device)


if __name__ == "__main__":
    start = time.time()
    main()
    end = time.time()
    h, m, s = (end - start) // 3600, (end - start) % 3600 // 60, (end - start) % 3600 % 60 
    print("Time used: {:.0f} hour, {:.0f} minute, {:.0f} second\n".format(h, m, s))