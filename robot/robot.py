from robot.robot_base import RobotBase
from importlib import import_module
from dataclasses import dataclass, field
from typing import List, Dict, Any
import torch

ROBOT_CLASS_PATH = {
    "panda": "robot.robot.Panda",
    "ur3": "robot.robot.UR3",
    "piper": "robot.robot.Piper",
}

class Robot(RobotBase):
    def __init__(self, robot_id: str):
        self.robot_id = robot_id.lower()
        self.const = self.get_robot_constants()
        super().__init__(self.const)

    def get_robot_constants(self) -> Dict[str, Any]:
        if self.robot_id not in ROBOT_CLASS_PATH:
            raise ValueError(f"Unsupported robot_id '{self.robot_id}'. Supported: {', '.join(ROBOT_CLASS_PATH)}")

        module_path, _, class_name = ROBOT_CLASS_PATH[self.robot_id].rpartition(".")
        mod = import_module(module_path)
        cls = getattr(mod, class_name)
        return cls().as_dict()
    
    def forward_kinematics_torch(self, theta_batch: torch.Tensor):
        if self.const["DH_TYPE"] == "modified":
            a = self.const["DH_A"]
            d = self.const["DH_D"]
            alpha = self.const["DH_ALPHA"]
            return super().forward_kinematics_torch_mdh(theta_batch, a, d, alpha)
        else:  # standard
            a = self.const["DH_A"]
            d = self.const["DH_D"]
            alpha = self.const["DH_ALPHA"]
            d_flange = self.const["D_FLANGE"]
            return super().forward_kinematics_torch_dh(theta_batch, a, d, alpha, d_flange)

@dataclass(frozen=True)
class Panda:
    MAX_JOINTS_UESD: List[float] = field(default_factory=lambda: [166, 101, 166, -4, 166, 215, 166])
    MIN_JOINTS_UESD: List[float] = field(default_factory=lambda: [-166, -101, -166, -176, -166, -1, -166])

    MAX_XYZRPY_AREA: List[float] = field(default_factory=lambda: [0.85, 0.85, 0.89, 180, 90, 180])
    MIN_XYZRPY_AREA: List[float] = field(default_factory=lambda: [-0.85, -0.85, -0.23, -180, -90, -180])

    RESTRICTED_AREA: bool = False

    DELTA_ANGLE_RANGE: float = 2.0

    MAX_XYZRPY: List[float] = field(default_factory=lambda: [0.07, 0.065, 0.03, 180, 6.1, 180])
    MIN_XYZRPY: List[float] = field(default_factory=lambda: [-0.073, -0.065, -0.028, -180, -6.1, -180])

    DOF: int = 7  
    
    DH_A: List[float] = field(default_factory=lambda: [0.0, 0.0, 0.0, 0.0825, -0.0825, 0.0, 0.088])
    DH_D: List[float] = field(default_factory=lambda: [0.333, 0.0, 0.316, 0.0, 0.384, 0.0, 0.0])
    DH_ALPHA: List[float] = field(default_factory=lambda: [
        0.0, -torch.pi/2, torch.pi/2, torch.pi/2, -torch.pi/2, torch.pi/2, torch.pi/2
    ])
    D_FLANGE: float = 0.107
    DH_TYPE: str = "standard"

    def as_dict(self) -> Dict[str, Any]:
        max_delta = [self.DELTA_ANGLE_RANGE] * self.DOF
        min_delta = [-self.DELTA_ANGLE_RANGE] * self.DOF
        return {
            "MAX_JOINTS_UESD": self.MAX_JOINTS_UESD,
            "MIN_JOINTS_UESD": self.MIN_JOINTS_UESD,
            "MAX_XYZRPY_AREA": self.MAX_XYZRPY_AREA,
            "MIN_XYZRPY_AREA": self.MIN_XYZRPY_AREA,
            "RESTRICTED_AREA": self.RESTRICTED_AREA,
            "DELTA_ANGLE_RANGE": self.DELTA_ANGLE_RANGE,
            "MAX_DELTA_ANGLE_RANGE": max_delta,
            "MIN_DELTA_ANGLE_RANGE": min_delta,
            "MAX_XYZRPY": self.MAX_XYZRPY,
            "MIN_XYZRPY": self.MIN_XYZRPY,
            "DOF": self.DOF,
            "DH_A": self.DH_A,
            "DH_D": self.DH_D,
            "DH_ALPHA": self.DH_ALPHA,
            "D_FLANGE": self.D_FLANGE,
            "DH_TYPE": self.DH_TYPE,
        }

@dataclass(frozen=True)
class UR3:
    restricted_area: bool = False
    delta_angle_range: float = 2.0  

    MAX_JOINTS_UESD: List[float] = field(default_factory=lambda: [180, 180, 180, 180, 180, 180])
    MIN_JOINTS_UESD: List[float] = field(default_factory=lambda: [-180, -180, -180, -180, -180, -180])

    MAX_XYZRPY_AREA: List[float] = field(default_factory=lambda: [0.59, 0.59, 0.73, 180, 90, 180])
    MIN_XYZRPY_AREA: List[float] = field(default_factory=lambda: [-0.59, -0.59, -0.43, -180, -90, -180])

    MAX_XYZRPY: List[float] = field(default_factory=lambda: [0.033, 0.034, 0.033, 180, 7.4, 180])
    MIN_XYZRPY: List[float] = field(default_factory=lambda: [-0.035, -0.034, -0.032, -180, -7.8, -180])

    DOF: int = 6
    
    DH_A: List[float] = field(default_factory=lambda: [0.0, -0.24365, -0.21325, 0.0, 0.0, 0.0])
    DH_D: List[float] = field(default_factory=lambda: [0.1519, 0.0, 0.0, 0.11235, 0.08535, 0.0819])
    DH_ALPHA: List[float] = field(default_factory=lambda: [
        torch.pi/2, 0.0, 0.0, torch.pi/2, -torch.pi/2, 0.0
    ])
    D_FLANGE: float = 0.0
    DH_TYPE: str = "standard"

    def as_dict(self) -> Dict[str, Any]:
        max_delta = [self.delta_angle_range] * self.DOF
        min_delta = [-self.delta_angle_range] * self.DOF
        return {
            "DOF": self.DOF,
            "RESTRICTED_AREA": self.restricted_area,
            "DELTA_ANGLE_RANGE": self.delta_angle_range,
            "MAX_JOINTS_UESD": self.MAX_JOINTS_UESD,
            "MIN_JOINTS_UESD": self.MIN_JOINTS_UESD,
            "MAX_XYZRPY_AREA": self.MAX_XYZRPY_AREA,
            "MIN_XYZRPY_AREA": self.MIN_XYZRPY_AREA,
            "MAX_DELTA_ANGLE_RANGE": max_delta,
            "MIN_DELTA_ANGLE_RANGE": min_delta,
            "MAX_XYZRPY": self.MAX_XYZRPY,
            "MIN_XYZRPY": self.MIN_XYZRPY,
            "DH_A": self.DH_A,
            "DH_D": self.DH_D,
            "DH_ALPHA": self.DH_ALPHA,
            "D_FLANGE": self.D_FLANGE,
            "DH_TYPE": self.DH_TYPE,
        }

@dataclass(frozen=True)
class Piper:
    restricted_area: bool = False
    delta_angle_range: float = 5.0  

    JOINTS_OFFSET: List[float] = field(default_factory=lambda: [0, -172.22, -102.78, 0, 0, 0])
    MAX_JOINTS: List[float] = field(default_factory=lambda: [154, 195, 0, 102, 75, 120])
    MIN_JOINTS: List[float] = field(default_factory=lambda: [-154, 0, -175, -102, -75, -120])

    MAX_JOINTS_RESTRICTED: List[float] = field(default_factory=lambda: [50, 155, -40, 50, 75, 90])
    MIN_JOINTS_RESTRICTED: List[float] = field(default_factory=lambda: [-50, 100, -140, -50, 20, -90])

    DOF: int = 6
    DH_A: List[float] = field(default_factory=lambda: [0.0, 0.0, 0.28503, -0.02198, 0.0, 0.0])
    DH_D: List[float] = field(default_factory=lambda: [0.123, 0.0, 0.0, 0.25075, 0.0, 0.091])
    DH_ALPHA: List[float] = field(default_factory=lambda: [
        0.0, -torch.pi/2, 0.0, torch.pi/2, -torch.pi/2, torch.pi/2
    ])
    D_FLANGE: float = 0.0
    DH_TYPE: str = "modified"

    def _calc_joint_used_and_area(self):
        if self.restricted_area:
            max_used = [self.MAX_JOINTS_RESTRICTED[i] + self.JOINTS_OFFSET[i] for i in range(self.DOF)]
            min_used = [self.MIN_JOINTS_RESTRICTED[i] + self.JOINTS_OFFSET[i] for i in range(self.DOF)]
            max_area = [0.74, 0.6, 0.71, 180, 90, 180]
            min_area = [-0.1, -0.6, -0.28, -180, -90, -180]
        else:
            max_used = [self.MAX_JOINTS[i] + self.JOINTS_OFFSET[i] for i in range(self.DOF)]
            min_used = [self.MIN_JOINTS[i] + self.JOINTS_OFFSET[i] for i in range(self.DOF)]
            max_area = [0.63, 0.63, 0.76, 180, 90, 180]
            min_area = [-0.63, -0.63, -0.33, -180, -90, -180]
        return max_used, min_used, max_area, min_area

    def _calc_delta_xyzrpy(self):
        if self.restricted_area:
            if self.delta_angle_range == 2.0:
                max_xyzrpy = [0.038, 0.036, 0.047, 180, 6.7, 180]
                min_xyzrpy = [-0.038, -0.036, -0.047, -180, -6.7, -180]
            elif self.delta_angle_range == 1.0:
                max_xyzrpy = [0.018, 0.018, 0.023, 160, 3.5, 160]
                min_xyzrpy = [-0.018, -0.018, -0.0235, -162, -3.5, -161]
            elif self.delta_angle_range == 5.0:
                max_xyzrpy = [0.093, 0.087, 0.118, 180, 15.8, 180]
                min_xyzrpy = [-0.092, -0.087, -0.117, -180, -16.2, -180]
            else:
                assert False, "delta_angle_range should be one of [1.0, 2.0, 5.0]"
        else:
            if self.delta_angle_range == 2.0:
                max_xyzrpy = [0.05, 0.05, 0.05, 180, 6.7, 180]
                min_xyzrpy = [-0.05, -0.05, -0.05, -180, -6.7, -180]
            elif self.delta_angle_range == 5.0:
                max_xyzrpy = [0.09, 0.09, 0.09, 180, 16.2, 180]
                min_xyzrpy = [-0.09, -0.09, -0.09, -180, -16.2, -180]
            else:
                assert False, "delta_angle_range should be one of [2.0, 5.0] in non-restricted area"
        return max_xyzrpy, min_xyzrpy

    def as_dict(self) -> Dict[str, Any]:
        max_used, min_used, max_area, min_area = self._calc_joint_used_and_area()
        max_delta = [self.delta_angle_range] * self.DOF
        min_delta = [-self.delta_angle_range] * self.DOF
        max_xyzrpy, min_xyzrpy = self._calc_delta_xyzrpy()

        return {
            "DOF": self.DOF,
            "RESTRICTED_AREA": self.restricted_area,
            "DELTA_ANGLE_RANGE": self.delta_angle_range,
            "JOINTS_OFFSET": self.JOINTS_OFFSET,
            "MAX_JOINTS": self.MAX_JOINTS,
            "MIN_JOINTS": self.MIN_JOINTS,
            "MAX_JOINTS_RESTRICTED": self.MAX_JOINTS_RESTRICTED,
            "MIN_JOINTS_RESTRICTED": self.MIN_JOINTS_RESTRICTED,
            "MAX_JOINTS_UESD": max_used,
            "MIN_JOINTS_UESD": min_used,
            "MAX_XYZRPY_AREA": max_area,
            "MIN_XYZRPY_AREA": min_area,
            "MAX_DELTA_ANGLE_RANGE": max_delta,
            "MIN_DELTA_ANGLE_RANGE": min_delta,
            "MAX_XYZRPY": max_xyzrpy,
            "MIN_XYZRPY": min_xyzrpy,
            "DH_A": self.DH_A,
            "DH_D": self.DH_D,
            "DH_ALPHA": self.DH_ALPHA,
            "D_FLANGE": self.D_FLANGE,
            "DH_TYPE": self.DH_TYPE,
        }
