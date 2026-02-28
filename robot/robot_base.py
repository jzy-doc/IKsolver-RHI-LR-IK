import torch

class RobotBase:
    def __init__(self, const):
        self.C = const
    
    def transform_mdh_torch(self, a, alpha, d, theta):
        """
        Batched MDH transformation matrix
        Input: (batch_size,)
        Output: (batch_size, 4, 4)
        """
        batch_size = theta.shape[0]
        cos_theta = torch.cos(theta)
        sin_theta = torch.sin(theta)
        cos_alpha = torch.cos(alpha)
        sin_alpha = torch.sin(alpha)

        T = torch.zeros(batch_size, 4, 4, device=theta.device)

        T[:, 0, 0] = cos_theta
        T[:, 0, 1] = -sin_theta
        T[:, 0, 2] = torch.zeros_like(theta)
        T[:, 0, 3] = a

        T[:, 1, 0] = sin_theta * cos_alpha
        T[:, 1, 1] = cos_theta * cos_alpha
        T[:, 1, 2] = -sin_alpha
        T[:, 1, 3] = -sin_alpha * d

        T[:, 2, 0] = sin_theta * sin_alpha
        T[:, 2, 1] = cos_theta * sin_alpha
        T[:, 2, 2] = cos_alpha
        T[:, 2, 3] = cos_alpha * d

        T[:, 3, 3] = 1.0
        return T
    
    def forward_kinematics_torch_mdh(self, theta_batch, a, d, alpha):
        """
        theta_batch: (batch_size, 6), a/d/alpha: list of 6 scalars
        return position, rpy (batch_size, 3)
        """
        batch_size = theta_batch.shape[0]
        T = torch.eye(4, device=theta_batch.device).unsqueeze(0).repeat(batch_size, 1, 1)

        a = torch.tensor(a, device=theta_batch.device).view(1, 6).expand(batch_size, -1)
        d = torch.tensor(d, device=theta_batch.device).view(1, 6).expand(batch_size, -1)
        alpha = torch.tensor(alpha, device=theta_batch.device).view(1, 6).expand(batch_size, -1)

        for i in range(6):
            Ti = self.transform_mdh_torch(a[:, i], alpha[:, i], d[:, i], theta_batch[:, i])
            T = torch.bmm(T, Ti)

        position = T[:, :3, 3]

        # rotation matrix to rpy (xyz order)
        rot_matrix = T[:, :3, :3]
        roll = torch.atan2(rot_matrix[:, 2, 1], rot_matrix[:, 2, 2])
        pitch = torch.asin(-rot_matrix[:, 2, 0].clamp(-1.0, 1.0))
        yaw = torch.atan2(rot_matrix[:, 1, 0], rot_matrix[:, 0, 0])
        rpy = torch.stack([roll, pitch, yaw], dim=1) * 180.0 / torch.pi
        return position, rpy

    def transform_dh_torch(self, a, alpha, d, theta):
        batch = theta.shape[0]
        ct = torch.cos(theta)
        st = torch.sin(theta)
        ca = torch.cos(alpha)
        sa = torch.sin(alpha)

        T = torch.zeros(batch, 4, 4, device=theta.device, dtype=theta.dtype)
        T[:, 0, 0] = ct
        T[:, 0, 1] = -st * ca
        T[:, 0, 2] =  st * sa
        T[:, 0, 3] = a * ct

        T[:, 1, 0] = st
        T[:, 1, 1] = ct * ca
        T[:, 1, 2] = -ct * sa
        T[:, 1, 3] = a * st

        T[:, 2, 0] = 0.0
        T[:, 2, 1] = sa
        T[:, 2, 2] = ca
        T[:, 2, 3] = d

        T[:, 3, 0] = 0.0
        T[:, 3, 1] = 0.0
        T[:, 3, 2] = 0.0
        T[:, 3, 3] = 1.0
        return T

    def forward_kinematics_torch_dh(self, theta_batch, a, d, alpha, d_flange=0):
        device = theta_batch.device
        dtype = theta_batch.dtype
        batch = theta_batch.shape[0]
        
        a = torch.as_tensor(a, device=device, dtype=dtype).view(1, 7).expand(batch, -1)
        d = torch.as_tensor(d, device=device, dtype=dtype).view(1, 7).expand(batch, -1)
        alpha = torch.as_tensor(alpha, device=device, dtype=dtype).view(1, 7).expand(batch, -1)

        T = torch.eye(4, device=device, dtype=dtype).unsqueeze(0).repeat(batch, 1, 1)
        for i in range(7):
            Ti = self.transform_dh_torch(a[:, i], alpha[:, i], d[:, i], theta_batch[:, i])
            T = torch.bmm(T, Ti)

        T_flange = torch.eye(4, device=device, dtype=dtype).unsqueeze(0).repeat(batch, 1, 1)
        T_flange[:, 2, 3] = d_flange
        T = torch.bmm(T, T_flange)

        position = T[:, :3, 3]  # (batch, 3)

        Rm = T[:, :3, :3]
        roll = torch.atan2(Rm[:, 2, 1], Rm[:, 2, 2])
        pitch = torch.asin((-Rm[:, 2, 0]).clamp(-1.0, 1.0))
        yaw = torch.atan2(Rm[:, 1, 0], Rm[:, 0, 0])

        rpy = torch.stack((roll, pitch, yaw), dim=1) * (180.0 / torch.pi)
        return position, rpy
    
    @staticmethod
    def _as_like(x: torch.Tensor, data, dtype=None):
        return torch.as_tensor(data, dtype=(dtype or x.dtype), device=x.device)

    def joints_cur_normalize_tensor(self, joints: torch.Tensor) -> torch.Tensor:
        min_t = self._as_like(joints, self.C["MIN_JOINTS_UESD"])
        max_t = self._as_like(joints, self.C["MAX_JOINTS_UESD"])
        return (joints - min_t) / (max_t - min_t)

    def joints_cur_denormalize_tensor(self, joints: torch.Tensor) -> torch.Tensor:
        min_t = self._as_like(joints, self.C["MIN_JOINTS_UESD"])
        max_t = self._as_like(joints, self.C["MAX_JOINTS_UESD"])
        return joints * (max_t - min_t) + min_t

    def delta_joints_normalize_tensor(self, delta_joints: torch.Tensor) -> torch.Tensor:
        min_t = self._as_like(delta_joints, [-self.C["DELTA_ANGLE_RANGE"]]*self.C["DOF"])
        max_t = self._as_like(delta_joints, [ self.C["DELTA_ANGLE_RANGE"]]*self.C["DOF"])
        return (delta_joints - min_t) / (max_t - min_t)

    def delta_joints_denormalize_tensor(self, delta_joints: torch.Tensor) -> torch.Tensor:
        min_t = self._as_like(delta_joints, [-self.C["DELTA_ANGLE_RANGE"]]*self.C["DOF"])
        max_t = self._as_like(delta_joints, [ self.C["DELTA_ANGLE_RANGE"]]*self.C["DOF"])
        return delta_joints * (max_t - min_t) + min_t

    def delta_xyzrpy_normalize_tensor(self, xyzrpy: torch.Tensor) -> torch.Tensor:
        min_t = self._as_like(xyzrpy, self.C["MIN_XYZRPY"])
        max_t = self._as_like(xyzrpy, self.C["MAX_XYZRPY"])
        den = torch.clamp(max_t - min_t, min=torch.finfo(xyzrpy.dtype).eps)
        return (xyzrpy - min_t) / den

    def delta_xyzrpy_denormalize_tensor(self, xyzrpy: torch.Tensor) -> torch.Tensor:
        min_t = self._as_like(xyzrpy, self.C["MIN_XYZRPY"])
        max_t = self._as_like(xyzrpy, self.C["MAX_XYZRPY"])
        return xyzrpy * (max_t - min_t) + min_t

    def xyzrpy_normalize_tensor(self, xyzrpy: torch.Tensor) -> torch.Tensor:
        min_t = self._as_like(xyzrpy, self.C["MIN_XYZRPY_AREA"])
        max_t = self._as_like(xyzrpy, self.C["MAX_XYZRPY_AREA"])
        den = torch.clamp(max_t - min_t, min=torch.finfo(xyzrpy.dtype).eps)
        return (xyzrpy - min_t) / den

    def xyzrpy_denormalize_tensor(self, xyzrpy: torch.Tensor) -> torch.Tensor:
        min_t = self._as_like(xyzrpy, self.C["MIN_XYZRPY_AREA"])
        max_t = self._as_like(xyzrpy, self.C["MAX_XYZRPY_AREA"])
        return xyzrpy * (max_t - min_t) + min_t
    