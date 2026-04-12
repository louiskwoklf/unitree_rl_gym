import torch

from isaacgym import gymtorch

from legged_gym.envs.base.legged_robot import LeggedRobot


class BruceStandRobot(LeggedRobot):
    LEG_JOINT_NAMES = (
        "hip_yaw_r",
        "hip_pitch_r",
        "hip_roll_r",
        "knee_pitch_r",
        "ankle_pitch_r",
        "hip_yaw_l",
        "hip_pitch_l",
        "hip_roll_l",
        "knee_pitch_l",
        "ankle_pitch_l",
    )

    def _init_buffers(self):
        super()._init_buffers()
        name_to_index = {name: i for i, name in enumerate(self.dof_names)}
        self.leg_indices = torch.tensor(
            [name_to_index[name] for name in self.LEG_JOINT_NAMES],
            dtype=torch.long,
            device=self.device,
        )

    def _expand_leg_actions(self, actions):
        # Keep the standing task focused on the legs while the arm joints hold
        # their default pose through the shared PD controller.
        full_actions = torch.zeros(
            actions.shape[0],
            self.num_dof,
            dtype=actions.dtype,
            device=actions.device,
        )
        full_actions[:, self.leg_indices] = actions
        return full_actions

    def _compute_torques(self, actions):
        return super()._compute_torques(self._expand_leg_actions(actions))

    def _get_noise_scale_vec(self, cfg):
        noise_vec = torch.zeros_like(self.obs_buf[0])
        self.add_noise = self.cfg.noise.add_noise
        noise_scales = self.cfg.noise.noise_scales
        noise_level = self.cfg.noise.noise_level

        dof_pos_start = 12
        dof_vel_start = dof_pos_start + self.num_dof
        action_start = dof_vel_start + self.num_dof

        noise_vec[:3] = noise_scales.lin_vel * noise_level * self.obs_scales.lin_vel
        noise_vec[3:6] = noise_scales.ang_vel * noise_level * self.obs_scales.ang_vel
        noise_vec[6:9] = noise_scales.gravity * noise_level
        noise_vec[9:12] = 0.0
        noise_vec[dof_pos_start:dof_vel_start] = (
            noise_scales.dof_pos * noise_level * self.obs_scales.dof_pos
        )
        noise_vec[dof_vel_start:action_start] = (
            noise_scales.dof_vel * noise_level * self.obs_scales.dof_vel
        )
        noise_vec[action_start:action_start + self.num_actions] = 0.0
        return noise_vec

    def _reset_dofs(self, env_ids):
        if len(env_ids) == 0:
            return

        self.dof_pos[env_ids] = self.default_dof_pos
        self.dof_vel[env_ids] = 0.0

        env_ids_int32 = env_ids.to(dtype=torch.int32)
        self.gym.set_dof_state_tensor_indexed(
            self.sim,
            gymtorch.unwrap_tensor(self.dof_state),
            gymtorch.unwrap_tensor(env_ids_int32),
            len(env_ids_int32),
        )

    def _reset_root_states(self, env_ids):
        if len(env_ids) == 0:
            return

        self.root_states[env_ids] = self.base_init_state
        self.root_states[env_ids, :3] += self.env_origins[env_ids]
        self.root_states[env_ids, 7:13] = 0.0

        env_ids_int32 = env_ids.to(dtype=torch.int32)
        self.gym.set_actor_root_state_tensor_indexed(
            self.sim,
            gymtorch.unwrap_tensor(self.root_states),
            gymtorch.unwrap_tensor(env_ids_int32),
            len(env_ids_int32),
        )

    def _resample_commands(self, env_ids):
        if len(env_ids) == 0:
            return
        self.commands[env_ids] = 0.0

    def _reward_alive(self):
        return 1.0
