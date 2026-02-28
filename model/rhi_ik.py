import torch
import torch.nn as nn

class HierarchicalUnit(torch.nn.Module):
    def __init__(self, input_dim, output_dim, net_arch=None, activate_fun=nn.ReLU()):
        super(HierarchicalUnit, self).__init__()
        
        if net_arch is None:
            # net_arch = [512, 256, 128]
            net_arch = [256, 128, 64]
            # net_arch = [128, 64]

        modules = [nn.Linear(input_dim, net_arch[0]), activate_fun]
        
        for idx in range(len(net_arch) - 1):
            modules.append(nn.Linear(net_arch[idx], net_arch[idx + 1]))
            modules.append(activate_fun)
        
        modules.append(nn.Linear(net_arch[-1], output_dim))
        modules.append(nn.Sigmoid())

        self.net = nn.Sequential(*modules)

    def forward(self, x):
        return self.net(x)

class ResidualBlock(nn.Module):
    def __init__(self, dim, activation=nn.ReLU()):
        super(ResidualBlock, self).__init__()
        self.block = nn.Sequential(
            nn.Linear(dim, dim),
            activation,
            nn.Linear(dim, dim)
        )
        self.activation = activation

    def forward(self, x):
        return self.activation(x + self.block(x))  

class ResidualHierarchicalUnit(nn.Module):
    def __init__(self, input_dim, output_dim, net_arch=None, activation=nn.ReLU()):
        super().__init__()

        if net_arch is None:
            net_arch = [512, 256, 128]
        
        self.input_layer = nn.Sequential(
            nn.Linear(input_dim, net_arch[0]),
            activation
        )
        
        layers = []
        for i in range(len(net_arch) - 1):
            in_dim = net_arch[i]
            next_dim = net_arch[i + 1]

            layers.append(ResidualBlock(in_dim, activation))
            layers.append(nn.Linear(in_dim, next_dim))
            layers.append(activation)
        
        self.res_blocks = nn.Sequential(*layers)
            
        self.output_layer = nn.Sequential(
            nn.Linear(net_arch[-1], output_dim),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.input_layer(x)
        x = self.res_blocks(x)
        return self.output_layer(x)

class PlainBlock(nn.Module):
    """
    与 ResidualBlock(dim) 参数规模一致的“无残差”版本：
    Linear(dim, dim) -> act -> Linear(dim, dim) -> act
    """
    def __init__(self, dim, activation=nn.ReLU()):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim),
            activation,
            nn.Linear(dim, dim),
            activation
        )

    def forward(self, x):
        return self.net(x)

class HierarchicalUnitNoResidual(nn.Module):
    def __init__(self, input_dim, output_dim, net_arch=None, activation=nn.ReLU()):
        super().__init__()
        if net_arch is None:
            net_arch = [256, 128, 128]  

        self.input_layer = nn.Sequential(
            nn.Linear(input_dim, net_arch[0]),
            activation
        )

        layers = []
        for i in range(len(net_arch) - 1):
            in_dim = net_arch[i]
            next_dim = net_arch[i + 1]

            layers.append(PlainBlock(in_dim, activation))
            layers.append(nn.Linear(in_dim, next_dim))
            layers.append(activation)

        self.blocks = nn.Sequential(*layers)

        self.output_layer = nn.Sequential(
            nn.Linear(net_arch[-1], output_dim),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.input_layer(x)
        x = self.blocks(x)
        return self.output_layer(x)

class HierarchicalInverseModel(nn.Module):
    def __init__(self, dof: int, task_dim: int = 6, net_arch=None):
        super().__init__()
        assert dof >= 1, "DOF must be >= 1"
        self.dof = dof
        self.task_dim = task_dim

        units = []
        for i in range(dof):
            in_dim = task_dim + i  
            print(f"Creating unit {i+1} with input dim {in_dim}")      
            units.append(HierarchicalUnitNoResidual(in_dim, 1, net_arch))
        self.units = nn.ModuleList(units)

    def forward(self, pos: torch.Tensor) -> torch.Tensor:
        qs = []
        for i, unit in enumerate(self.units):
            if i == 0:
                x = pos
            else:
                x = torch.cat([*qs, pos], dim=1)
            q_i = unit(x)          
            qs.append(q_i)
        return torch.cat(qs, dim=1)  

class ResidualHierarchicalInverseModel(nn.Module):
    def __init__(self, dof: int, task_dim: int = 6, net_arch=None):
        super().__init__()
        assert dof >= 1, "DOF must be >= 1"
        self.dof = dof
        self.task_dim = task_dim

        units = []
        for i in range(dof):
            in_dim = task_dim + i  
            print(f"Creating unit {i+1} with input dim {in_dim}")      
            units.append(ResidualHierarchicalUnit(in_dim, 1, net_arch))
        self.units = nn.ModuleList(units)

    def forward(self, pos: torch.Tensor) -> torch.Tensor:
        qs = []
        for i, unit in enumerate(self.units):
            if i == 0:
                x = pos
            else:
                x = torch.cat([*qs, pos], dim=1)
            q_i = unit(x)          
            qs.append(q_i)
        return torch.cat(qs, dim=1)  



 
