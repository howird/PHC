import torch
import numpy as np
from phc.env.tasks.humanoid_amp import HumanoidAMP
from phc.utils import torch_utils
from isaacgym.torch_utils import *

class HumanoidAMPHeading(HumanoidAMP):
    def __init__(
        self, cfg, sim_params, physics_engine, device_type, device_id, headless
    ):
        super().__init__(
            cfg, sim_params, physics_engine, device_type, device_id, headless
        )
        
        self.distance_threshold = cfg["env"]["heading_reward"]["distance_threshold"]
        self.target_range = cfg["env"]["heading_reward"]["target_range"]
        self.termination_dist = self.cfg["env"]["heading_termination_distance"]
        
        self.pos_reward_scale = cfg["env"]["heading_reward"]["pos_reward_scale"]
        self.speed_reward_scale = cfg["env"]["heading_reward"]["speed_reward_scale"]

        self.pos_reward_weight = cfg["env"]["heading_reward"]["pos_reward_weight"]
        self.speed_reward_weight = cfg["env"]["heading_reward"]["speed_reward_weight"]
        self.dir_reward_weight = cfg["env"]["heading_reward"]["dir_reward_weight"]
        
        self.target_speed = cfg["env"]["heading_reward"]["target_speed"]

        self.target_pos = torch.zeros(
            (self.num_envs, 2), 
            device=self.device
        )
        
        # Randomize initial targets
        self._randomize_targets()

    def _randomize_targets(self, env_ids=None):
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)
        
        self.target_pos[env_ids, 0] = torch.rand(
            len(env_ids), 
            device=self.device
        ) * (self.target_range["x"][1] - self.target_range["x"][0]) + self.target_range["x"][0]
        
        self.target_pos[env_ids, 1] = torch.rand(
            len(env_ids), 
            device=self.device
        ) * (self.target_range["y"][1] - self.target_range["y"][0]) + self.target_range["y"][0]

    def _compute_reward(self, actions):
        root_pos = self._rigid_body_pos[:, 0, :2]
        root_rot = self._rigid_body_rot[:, 0, :]
        root_vel = self._rigid_body_vel[:, 0, :2]
        
        target_vec = self.target_pos[:, :2] - root_pos
        distance_to_target = torch.norm(target_vec, dim=-1)
        pos_reward = torch.exp(-self.pos_reward_scale * (distance_to_target**2))
        
        heading_rot = torch_utils.calc_heading_quat(root_rot)
        desired_dir = torch.atan2(target_vec[:, 1], target_vec[:, 0])
        dir_diff = torch.abs(
            torch_utils.quat_to_angle_axis(heading_rot)[0] - desired_dir
        )
        dir_reward = (torch.cos(dir_diff) + 1)/2

        target_dir = target_vec / (torch.norm(target_vec, dim=-1, keepdim=True) + 1e-8)
        speed_proj = torch.sum(root_vel * target_dir, dim=-1)
        speed_reward = torch.exp(-self.speed_reward_scale * torch.maximum(torch.tensor(0), self.target_speed - speed_proj))
        
        self.rew_buf[:] = self.pos_reward_weight*pos_reward + dir_reward + self.speed_reward_weight*speed_reward
        
        # Store raw rewards for debugging
        self.reward_raw = torch.stack([pos_reward, dir_reward, speed_reward], dim=-1)
        
        # Reset condition
        self.reset_buf = torch.where(
            distance_to_target < self.distance_threshold, 
            torch.ones_like(self.reset_buf), 
            self.reset_buf
        )
        
        return

    def _compute_task_obs(self, env_ids=None):
        if env_ids is None:
            root_pos = self._rigid_body_pos[:, 0, :2]
            root_vel = self._rigid_body_vel[:, 0, :2]
            target_pos = self.target_pos[:, :2]
        else:
            root_pos = self._rigid_body_pos[env_ids, 0, :2]
            root_vel = self._rigid_body_vel[env_ids, 0, :2]
            target_pos = self.target_pos[env_ids, :2]
        
        # Target position vector
        target_vec = target_pos - root_pos
        target_vec_norm = target_vec / (torch.norm(target_vec, dim=-1, keepdim=True) + 1e-8)
        
        # Current velocity projection onto target direction
        target_dir = target_vec / (torch.norm(target_vec, dim=-1, keepdim=True) + 1e-8)
        vel_proj = torch.sum(root_vel * target_dir, dim=-1, keepdim=True)
        
        # Combine observations
        return torch.cat([target_vec_norm, vel_proj], dim=-1)

    def get_task_obs_size(self):
        return 3  # 2D normalized target vector + 1D velocity projection

    def reset(self, env_ids=None):
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)
        self._randomize_targets(env_ids)
        super().reset(env_ids)

    def _compute_reset(self):
        # Check if episode has reached max length
        pass_time_max = self.progress_buf >= self.max_episode_length - 1
        
        # Check distance to target
        root_pos = self._rigid_body_pos[:, 0, :2]
        distance_to_target = torch.norm(self.target_pos[:, :2] - root_pos, dim=-1)
        
        reached_target = distance_to_target < self.termination_dist
        reset = torch.logical_or(pass_time_max, reached_target)
        
        # Set reset and terminate buffers
        self.reset_buf[:] = reset.float()
        self._terminate_buf[:] = reset.float()
