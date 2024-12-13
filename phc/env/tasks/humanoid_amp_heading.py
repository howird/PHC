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

        # Speed variation parameters
        self.target_speed = cfg["env"]["heading_reward"]["target_speed"]

        self.distance_threshold = cfg["env"]["heading_reward"]["distance_threshold"]
        self.target_range = cfg["env"]["heading_reward"]["target_range"]
        self.termination_dist = self.cfg["env"]["heading_termination_distance"]

        self.pos_reward_scale = cfg["env"]["heading_reward"]["pos_reward_scale"]
        self.speed_reward_scale = cfg["env"]["heading_reward"]["speed_reward_scale"]

        self.pos_reward_weight = cfg["env"]["heading_reward"]["pos_reward_weight"]
        self.speed_reward_weight = cfg["env"]["heading_reward"]["speed_reward_weight"]
        self.dir_reward_weight = cfg["env"]["heading_reward"]["dir_reward_weight"]

        self.target_pos = torch.zeros((self.num_envs, 2), device=self.device)
        self.speed_type = self.target_speed["type"]

        # Randomize initial targets and speeds
        self._randomize_targets_and_speeds()

    def _randomize_targets_and_speeds(self, env_ids=None):
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)

        # Randomize target positions (existing logic)
        self.target_pos[env_ids, 0] = (
            torch.rand(len(env_ids), device=self.device)
            * (self.target_range["x"][1] - self.target_range["x"][0])
            + self.target_range["x"][0]
        )
        self.target_pos[env_ids, 1] = (
            torch.rand(len(env_ids), device=self.device)
            * (self.target_range["y"][1] - self.target_range["y"][0])
            + self.target_range["y"][0]
        )

        if self.speed_type == "constant":
            return

        self.current_target_speeds = torch.zeros(self.num_envs, device=self.device)
        if self.speed_type == "beta":
            min_speed = self.target_speed["min_speed"]
            max_speed = self.target_speed["max_speed"]
            speeds = (
                torch.distributions.Beta(
                    torch.tensor([2]).to(self.device), torch.tensor([5]).to(self.device)
                )
                .sample((len(env_ids),))
                .squeeze()
            )

            speeds = min_speed + speeds * (max_speed - min_speed)

        elif self.speed_type == "normal":
            min_speed = self.target_speed["min_speed"]
            max_speed = self.target_speed["max_speed"]
            mean_speed = self.target_speed.get("mean_speed", 1.0)
            std_speed = self.target_speed.get("std_speed", 0.3)

            # Generate normal distribution
            speeds = torch.clamp(
                torch.normal(
                    mean=mean_speed * torch.ones(len(env_ids), device=self.device),
                    std=std_speed,
                ),
                min=min_speed,
                max=max_speed,
            )

        else:
            min_speed = self.target_speed["min_speed"]
            max_speed = self.target_speed["max_speed"]
            speeds = (
                torch.rand(len(env_ids), device=self.device) * (max_speed - min_speed)
                + min_speed
            )

        self.current_target_speeds[env_ids] = speeds

    def _compute_reward(self, actions):
        root_pos = self._rigid_body_pos[:, 0, :2]
        root_rot = self._rigid_body_rot[:, 0, :]
        root_vel = self._rigid_body_vel[:, 0, :2]

        target_vec = self.target_pos[:, :2] - root_pos
        distance_to_target = torch.norm(target_vec, dim=-1)

        pos_reward = torch.exp(-self.pos_reward_scale * (distance_to_target**2))

        heading_rot = torch_utils.calc_heading_quat(root_rot)
        desired_heading = torch.atan2(target_vec[:, 1], target_vec[:, 0])
        heading_diff = torch.abs(
            torch_utils.quat_to_angle_axis(heading_rot)[0] - desired_heading
        )
        heading_reward = (torch.cos(heading_diff) + 1) / 2

        target_dir = target_vec / (torch.norm(target_vec, dim=-1, keepdim=True) + 1e-8)
        vel_proj = torch.sum(root_vel * target_dir, dim=-1)
        speed_reward = torch.exp(
            -self.speed_reward_scale
            * torch.maximum(
                torch.tensor(0),
                (
                    self.target_speed["constant_speed"]
                    if self.speed_type == "constant"
                    else self.current_target_speeds
                )
                - vel_proj,
            )
        )

        self.rew_buf[:] = (
            self.pos_reward_weight * pos_reward
            + self.dir_reward_weight * heading_reward
            + self.speed_reward_weight * speed_reward
        )

        # Store raw rewards for debugging
        self.reward_raw = torch.stack(
            [pos_reward, heading_reward, speed_reward], dim=-1
        )

        # Reset condition
        self.reset_buf = torch.where(
            distance_to_target < self.distance_threshold,
            torch.ones_like(self.reset_buf),
            self.reset_buf,
        )

        return

    def _compute_task_obs(self, env_ids=None):
        if env_ids is None:
            root_pos = self._rigid_body_pos[:, 0, :2]
            root_vel = self._rigid_body_vel[:, 0, :2]
            target_pos = self.target_pos[:, :2]
            if self.speed_type != "constant":
                target_speeds = self.current_target_speeds
        else:
            root_pos = self._rigid_body_pos[env_ids, 0, :2]
            root_vel = self._rigid_body_vel[env_ids, 0, :2]
            target_pos = self.target_pos[env_ids, :2]
            if self.speed_type != "constant":
                target_speeds = self.current_target_speeds[env_ids]

        # Target position vector
        target_vec = target_pos - root_pos
        target_vec_norm = target_vec / (
            torch.norm(target_vec, dim=-1, keepdim=True) + 1e-8
        )

        # Current velocity projection onto target direction
        target_dir = target_vec / (torch.norm(target_vec, dim=-1, keepdim=True) + 1e-8)
        vel_proj = torch.sum(root_vel * target_dir, dim=-1, keepdim=True)

        # Combine observations: normalized target vector, velocity projection, target speed
        if self.speed_type == "constant":
            return torch.cat([target_vec_norm, vel_proj], dim=-1)
        else:
            return torch.cat([target_vec_norm, vel_proj, target_speeds[:, None]], dim=-1)

    def get_task_obs_size(self):
        # 2D normalized target vector + 1D velocity projection + 1D target speed
        return 3 if self.speed_type == "constant" else 4

    def reset(self, env_ids=None):
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)
        self._randomize_targets_and_speeds(env_ids)
        super().reset(env_ids)

    def _compute_reset(self):
        # Existing reset logic with speed variation
        pass_time_max = self.progress_buf >= self.max_episode_length - 1

        root_pos = self._rigid_body_pos[:, 0, :2]
        distance_to_target = torch.norm(self.target_pos[:, :2] - root_pos, dim=-1)

        reached_target = distance_to_target < self.termination_dist
        reset = torch.logical_or(pass_time_max, reached_target)

        self.reset_buf[:] = reset.float()
        self._terminate_buf[:] = reset.float()
