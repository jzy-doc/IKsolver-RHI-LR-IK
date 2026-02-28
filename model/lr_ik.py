import torch
import torch.nn as nn

class DynamicModel(nn.Module):
    def __init__(self, input_dim=12, joint_output_dim=6, net_arch=None, num_heads=2, num_layers=1, dim_feedforward=128):
    
        super(DynamicModel, self).__init__()

        if net_arch is None:
            net_arch = [1024, 512, 512, 256]  
            
        encoder_modules = [nn.Linear(input_dim, net_arch[0]), nn.ReLU()]
        for idx in range(len(net_arch) - 1):
            encoder_modules.append(nn.Linear(net_arch[idx], net_arch[idx + 1]))
            encoder_modules.append(nn.ReLU())
        self.encoder = nn.Sequential(*encoder_modules)

        self.joint_decoder = nn.Sequential(
            nn.Linear(net_arch[-1], 128),
            nn.ReLU(),
            nn.Linear(128, joint_output_dim),
            nn.Sigmoid()  
        )

    def forward(self, x):
        encoded = self.encoder(x)  
        joint_output = self.joint_decoder(encoded)  
        return joint_output





