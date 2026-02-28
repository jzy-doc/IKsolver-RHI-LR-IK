import sys
sys.path.append("..")

import math

import torch
import torch.nn as nn

from model.rhi_ik import HierarchicalInverseModel


# def init_parameters_uniform(net, bias_fill_zeros=False):
#     """
#     Layer (activate: ReLu): use kaiming initialization
#     Layer (activate: Sigmoid): use xavier initialization
#     """
#     for layer in net.modules():
#         if isinstance(layer, nn.Linear):
#             nn.init.kaiming_uniform_(layer.weight, nonlinearity='relu')
#             if layer.bias is not None:
#                 if bias_fill_zeros:
#                     nn.init.constant_(layer.bias, 0)
#                 else:
#                     fan_in, _ = nn.init._calculate_fan_in_and_fan_out(layer.weight)
#                     bound = 1 / math.sqrt(fan_in)
#                     nn.init.uniform_(layer.bias, -bound, bound)
    
#     for layer in net.children():
#         # print(layer[-2])
#         nn.init.xavier_uniform_(layer[-2].weight, gain=nn.init.calculate_gain('sigmoid'))
#         # if layer[-2].bias is not None:
#         #     fan_in, _ = nn.init._calculate_fan_in_and_fan_out(layer[-2].weight)
#         #     bound = 1 / math.sqrt(fan_in)
#         #     nn.init.uniform_(layer[-2].bias, -bound, bound)

def init_parameters_uniform(net, bias_fill_zeros=False):
    """
    Layer (activation: ReLU): use kaiming initialization.
    Layer (activation: Sigmoid): use xavier initialization for the preceding layer.
    """
    for layer in net.modules():
        if isinstance(layer, nn.Linear):
            nn.init.kaiming_uniform_(layer.weight, nonlinearity='relu')
            if layer.bias is not None:
                if bias_fill_zeros:
                    nn.init.constant_(layer.bias, 0)
                else:
                    fan_in, _ = nn.init._calculate_fan_in_and_fan_out(layer.weight)
                    bound = 1 / math.sqrt(fan_in)
                    nn.init.uniform_(layer.bias, -bound, bound)
    
    for layer in net.children():
        if isinstance(layer, nn.Sequential):
            last_linear = None
            for sub_layer in reversed(layer):
                if isinstance(sub_layer, nn.Linear):
                    last_linear = sub_layer
                    break
            if last_linear:
                nn.init.xavier_uniform_(last_linear.weight, gain=nn.init.calculate_gain('sigmoid'))
                if last_linear.bias is not None:
                    fan_in, _ = nn.init._calculate_fan_in_and_fan_out(last_linear.weight)
                    bound = 1 / math.sqrt(fan_in)
                    nn.init.uniform_(last_linear.bias, -bound, bound)


def init_parameters_normal(net, bias_fill_zeros=True):
    """
    Layer (activate: ReLu): use kaiming initialization
    Layer (activate: Sigmoid): use xavier initialization
    """
    for layer in net.modules():
        if isinstance(layer, nn.Linear):
            nn.init.kaiming_normal_(layer.weight, nonlinearity='relu')
            if layer.bias is not None:
                if bias_fill_zeros:
                    nn.init.constant_(layer.bias, 0)
                else:
                    fan_in, _ = nn.init._calculate_fan_in_and_fan_out(layer.weight)
                    bound = 1 / math.sqrt(fan_in)
                    nn.init.uniform_(layer.bias, -bound, bound)
    
    for layer in net.children():
        # print(layer[-2])
        nn.init.xavier_normal_(layer[-2].weight, gain=nn.init.calculate_gain('sigmoid'))
        # if layer[-2].bias is not None:
        #     fan_in, _ = nn.init._calculate_fan_in_and_fan_out(layer[-2].weight)
        #     bound = 1 / math.sqrt(fan_in)
        #     nn.init.uniform_(layer[-2].bias, -bound, bound)

def init_extra_layers_parameters(extra_layers, bias_fill_zeros=False):
    for layer in extra_layers.modules():
        if isinstance(layer, nn.Linear):
            nn.init.kaiming_uniform_(layer.weight, nonlinearity='relu')
            if layer.bias is not None:
                if bias_fill_zeros:
                    nn.init.constant_(layer.bias, 0)
                else:
                    fan_in, _ = nn.init._calculate_fan_in_and_fan_out(layer.weight)
                    bound = 1 / math.sqrt(fan_in)
                    nn.init.uniform_(layer.bias, -bound, bound)

    last_linear = None
    for layer in reversed(list(extra_layers.modules())):
        if isinstance(layer, nn.Linear):
            last_linear = layer
            break
    if last_linear:
        nn.init.xavier_uniform_(last_linear.weight, gain=nn.init.calculate_gain('sigmoid'))
        if last_linear.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(last_linear.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(last_linear.bias, -bound, bound)

    print("Extra layers parameters initialized.")
    
def init_parameters_hierarchicalunit(net, bias_fill_zeros=False):
    """
    Layer (activate: ReLu): use kaiming initialization
    Layer (activate: Sigmoid): use xavier initialization
    """
    for layer in net.modules():
        if isinstance(layer, nn.Linear):
            nn.init.kaiming_uniform_(layer.weight, nonlinearity='relu')
            if layer.bias is not None:
                if bias_fill_zeros:
                    nn.init.constant_(layer.bias, 0)
                else:
                    fan_in, _ = nn.init._calculate_fan_in_and_fan_out(layer.weight)
                    bound = 1 / math.sqrt(fan_in)
                    nn.init.uniform_(layer.bias, -bound, bound)

    for layer in net.modules():
        if isinstance(layer, nn.Sequential):
            if len(layer) >= 2 and isinstance(layer[-2], nn.Linear):
                nn.init.xavier_uniform_(
                    layer[-2].weight,
                    gain=nn.init.calculate_gain('sigmoid')
                )