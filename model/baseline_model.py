import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Callable, Optional, List

class RNN_MLP(nn.Module):
    def __init__(self, in_dim: int, hidden: int = 256, out_dim: int = 256, act=nn.ReLU()):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), act,
            nn.Linear(hidden, out_dim), act
        )
    def forward(self, x):
        return self.net(x)

class IKNetStyleGRU(nn.Module):
    def __init__(
        self,
        pose_dim: int = 6,
        n_joints: int = 6,
        hidden: int = 256,
        rnn_layers: int = 1,
        in_width: int = 256,
        out_width: int = 256,
        out_activation: Optional[nn.Module] = nn.Sigmoid(),
        project_fn: Optional[Callable[[torch.Tensor, torch.Tensor], torch.Tensor]] = None
    ):
        super().__init__()
        self.pose_dim   = pose_dim
        self.n_joints   = n_joints
        self.hidden     = hidden
        self.project_fn = project_fn

        self.gru = nn.GRU(
            input_size=in_width,
            hidden_size=hidden,
            num_layers=rnn_layers,
            batch_first=True,
            dropout=0.1 if rnn_layers > 1 else 0
        )

        self.mlp_in = nn.ModuleList()
        for i in range(n_joints):
            in_dim_i = pose_dim + i 
            self.mlp_in.append(RNN_MLP(in_dim=in_dim_i, hidden=in_width, out_dim=in_width))

        self.mlp_out = nn.ModuleList()
        for _ in range(n_joints):
            self.mlp_out.append(RNN_MLP(in_dim=hidden, hidden=out_width, out_dim=out_width))
        self.heads = nn.ModuleList([nn.Linear(out_width, 1) for _ in range(n_joints)])
        self.out_act = out_activation  # e.g., nn.Sigmoid() 或 nn.Tanh() 或 None

    def _maybe_project_pose(self, pose: torch.Tensor, q_prefix: torch.Tensor) -> torch.Tensor:
        if self.project_fn is None:
            return pose
        return self.project_fn(pose, q_prefix)

    def forward(self, pose: torch.Tensor, q_init: Optional[torch.Tensor] = None) -> torch.Tensor:
        B = pose.size(0)
        device = pose.device

        q_preds: List[torch.Tensor] = []
        h_t = torch.zeros(self.gru.num_layers, B, self.hidden, device=device)

        for i in range(self.n_joints):
            if i == 0:
                q_prefix = torch.empty(B, 0, device=device)
            else:
                q_prefix = torch.cat(q_preds, dim=1) 

            pose_i = self._maybe_project_pose(pose, q_prefix) 

            x_i = torch.cat([pose_i, q_prefix], dim=1) 

            feat_i = self.mlp_in[i](x_i) 

            gru_in = feat_i.unsqueeze(1)  
            gru_out, h_t = self.gru(gru_in, h_t) 

            z_i = self.mlp_out[i](gru_out[:, -1, :]) 
            qi  = self.heads[i](z_i)                 
            if self.out_act is not None:
                qi = self.out_act(qi)

            q_preds.append(qi)

        return torch.cat(q_preds, dim=1) 
    
class MLP(torch.nn.Module):
    def __init__(self, input_dim=6, output_dim=6, net_arch=None):
        super(MLP, self).__init__()
        
        if net_arch is None:
            net_arch = [512, 1024, 512, 256, 128]

        net_modules = [nn.Linear(input_dim, net_arch[0]), nn.ReLU()]
        
        for idx in range(len(net_arch) - 1):
            net_modules.append(nn.Linear(net_arch[idx], net_arch[idx + 1]))
            net_modules.append(nn.ReLU())
        
        net_modules.append(nn.Linear(net_arch[-1], output_dim))
        net_modules.append(nn.Sigmoid())

        self.net = nn.Sequential(*net_modules)

    def forward(self, x):
        return self.net(x)

class CNNModel(nn.Module):
    def __init__(self, input_dim=6, output_dim=6):
        super(CNNModel, self).__init__()
        self.conv1d = nn.Conv1d(in_channels=1, out_channels=512, kernel_size=3, padding=1)  
        self.relu1 = nn.ReLU()
        
        self.pool1 = nn.MaxPool1d(kernel_size=2, stride=2)
        self.downsample1 = nn.Conv1d(in_channels=512, out_channels=256, kernel_size=3, padding=1)
        self.relu2 = nn.ReLU()
        
        self.flatten = nn.Flatten()
        
        self.dense1 = nn.Linear(256 * 3, 512)  
        self.relu5 = nn.ReLU()
        self.dense2 = nn.Linear(512, 256)
        self.relu6 = nn.ReLU()
        self.dense3 = nn.Linear(256, 512)
        self.relu7 = nn.ReLU()
        self.dense4 = nn.Linear(512, 256)
        self.relu8 = nn.ReLU()
        
        self.batch_norm = nn.BatchNorm1d(256)
        self.dense5 = nn.Linear(256, output_dim)
        
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        if x.dim() == 2:
            x = x.unsqueeze(1) 
        
        x = self.conv1d(x)
        x = self.relu1(x)
        
        x = self.pool1(x)
        x = self.downsample1(x)
        x = self.relu2(x)
        
        x = self.flatten(x)
        
        x = self.dense1(x)
        x = self.relu5(x)
        x = self.dense2(x)
        x = self.relu6(x)
        x = self.dense3(x)
        x = self.relu7(x)
        x = self.dense4(x)
        x = self.relu8(x)
        x = self.batch_norm(x)
        x = self.dense5(x)
        x = self.sigmoid(x)
        
        return x