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
        
        # Reward configuration from config file
        self.pos_reward_scale = cfg["env"]["heading_reward"]["pos_reward_scale"]
        self.heading_reward_scale = cfg["env"]["heading_reward"]["heading_reward_scale"]
        self.distance_threshold = cfg["env"]["heading_reward"]["distance_threshold"]
        
        # Target range configuration from config file
        self.target_range = cfg["env"]["heading_reward"]["target_range"]
        self.termination_dist = self.cfg["env"]["heading_termination_distance"]
        
        # Initialize target positions for all environments
        self.target_pos = torch.zeros(
            (self.num_envs, 3), 
            device=self.device
        )
        
        # Randomize initial targets
        self._randomize_targets()

    def _randomize_targets(self, env_ids=None):
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)
        assert torch.all(env_ids >= 0) and torch.all(env_ids < self.num_envs), "Invalid env_ids"
        
        self.target_pos[env_ids, 0] = torch.rand(
            len(env_ids), 
            device=self.device
        ) * (self.target_range["x"][1] - self.target_range["x"][0]) + self.target_range["x"][0]
        
        self.target_pos[env_ids, 1] = torch.rand(
            len(env_ids), 
            device=self.device
        ) * (self.target_range["y"][1] - self.target_range["y"][0]) + self.target_range["y"][0]
        
        self.target_pos[env_ids, 2] = torch.tensor(
            self.target_range["z"][0], 
            device=self.device
        )

    def _compute_reward(self, actions):
        root_pos = self._rigid_body_pos[:, 0, :2]
        root_rot = self._rigid_body_rot[:, 0, :]
        
        target_vec = self.target_pos[:, :2] - root_pos
        distance_to_target = torch.norm(target_vec, dim=-1)
        
        pos_reward = torch.exp(-self.pos_reward_scale * distance_to_target**2)
        
        heading_rot = torch_utils.calc_heading_quat(root_rot)
        desired_heading = torch.atan2(target_vec[:, 1], target_vec[:, 0])
        desired_heading_quat = quat_from_angle_axis(
            desired_heading, 
            torch.tensor([0.0, 0.0, 1.0], device=self.device)
        )
        
        heading_diff = torch_utils.quat_to_angle_axis(
            quat_mul(heading_rot, quat_conjugate(desired_heading_quat))
        )[0]
        
        heading_reward = torch.cos(heading_diff)
        
        self.rew_buf[:] = pos_reward + self.heading_reward_scale * heading_reward
        
        self.reward_raw = torch.stack([pos_reward, heading_reward], dim=-1)
        
        self.reset_buf = torch.where(
            distance_to_target < self.distance_threshold, 
            torch.ones_like(self.reset_buf), 
            self.reset_buf
        )
        
        return

    def _compute_task_obs(self, env_ids=None):
        if env_ids is None:
            root_pos = self._rigid_body_pos[:, 0, :2]
            target_pos = self.target_pos[:, :2]
        else:
            root_pos = self._rigid_body_pos[env_ids, 0, :2]
            target_pos = self.target_pos[env_ids, :2]
        
        target_vec = target_pos - root_pos
        target_vec_norm = target_vec / (torch.norm(target_vec, dim=-1, keepdim=True) + 1e-8)
        
        return target_vec_norm

    def get_task_obs_size(self):
        return 2 # Normalized 2D vector to target

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
