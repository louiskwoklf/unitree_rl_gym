import torch

from isaacgym import gymtorch

from legged_gym.envs.base.legged_robot import LeggedRobot


class BruceRobot(LeggedRobot):
    @staticmethod
    def _format_body_names(body_names):
        return ", ".join(body_names) if body_names else "<none>"

    def _expand_leg_actions(self, actions):
        full_actions = torch.zeros(
            actions.shape[0],
            self.num_dof,
            dtype=actions.dtype,
            device=actions.device,
        )
        full_actions[:, self.leg_indices] = actions
        return full_actions

    def _compute_torques(self, actions):
        # Keep the action space focused on the legs for jump training. The arm
        # joints stay at their nominal pose to reduce search-space noise while
        # Bruce learns a symmetric crouch-push-land cycle.
        return super()._compute_torques(self._expand_leg_actions(actions))

    def _reset_dofs(self, env_ids):
        if len(env_ids) == 0:
            return

        # Bruce's nominal stance is compact enough that the base class
        # 0.5x..1.5x multiplicative reset knocks the feet into the ground or
        # leaves them hanging above it. Start close to the calibrated pose.
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

        # Bruce also needs quiet resets. The shared base class injects random
        # root velocity, which is large for this small robot and soft controller.
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

    def _get_noise_scale_vec(self, cfg):
        """Sets a noise vector that matches Bruce's custom observation layout."""
        noise_vec = torch.zeros_like(self.obs_buf[0])
        self.add_noise = self.cfg.noise.add_noise
        noise_scales = self.cfg.noise.noise_scales
        noise_level = self.cfg.noise.noise_level

        dof_pos_start = 9
        dof_vel_start = dof_pos_start + self.num_dof
        action_start = dof_vel_start + self.num_dof
        contact_start = action_start + self.num_actions

        noise_vec[:3] = noise_scales.lin_vel * noise_level * self.obs_scales.lin_vel
        noise_vec[3:6] = noise_scales.ang_vel * noise_level * self.obs_scales.ang_vel
        noise_vec[6:9] = noise_scales.gravity * noise_level
        noise_vec[dof_pos_start:dof_vel_start] = (
            noise_scales.dof_pos * noise_level * self.obs_scales.dof_pos
        )
        noise_vec[dof_vel_start:action_start] = (
            noise_scales.dof_vel * noise_level * self.obs_scales.dof_vel
        )
        noise_vec[action_start:contact_start] = 0.0
        noise_vec[contact_start:] = 0.0
        return noise_vec

    def _init_foot(self):
        self.feet_num = len(self.feet_indices)

        rigid_body_state = self.gym.acquire_rigid_body_state_tensor(self.sim)
        self.rigid_body_states = gymtorch.wrap_tensor(rigid_body_state)
        self.rigid_body_states_view = self.rigid_body_states.view(self.num_envs, -1, 13)
        self.feet_state = self.rigid_body_states_view[:, self.feet_indices, :]
        self.feet_pos = self.feet_state[:, :, :3]
        self.feet_vel = self.feet_state[:, :, 7:10]

    def _init_buffers(self):
        super()._init_buffers()
        self._init_foot()

        name_to_index = {name: i for i, name in enumerate(self.dof_names)}
        self.hip_indices = torch.tensor(
            [
                name_to_index["hip_yaw_r"],
                name_to_index["hip_roll_r"],
                name_to_index["hip_yaw_l"],
                name_to_index["hip_roll_l"],
            ],
            dtype=torch.long,
            device=self.device,
        )
        self.leg_indices = torch.tensor(
            [
                name_to_index["hip_yaw_r"],
                name_to_index["hip_pitch_r"],
                name_to_index["hip_roll_r"],
                name_to_index["knee_pitch_r"],
                name_to_index["ankle_pitch_r"],
                name_to_index["hip_yaw_l"],
                name_to_index["hip_pitch_l"],
                name_to_index["hip_roll_l"],
                name_to_index["knee_pitch_l"],
                name_to_index["ankle_pitch_l"],
            ],
            dtype=torch.long,
            device=self.device,
        )
        self.arm_indices = torch.tensor(
            [
                name_to_index["shoulder_pitch_r"],
                name_to_index["shoulder_roll_r"],
                name_to_index["elbow_pitch_r"],
                name_to_index["shoulder_pitch_l"],
                name_to_index["shoulder_roll_l"],
                name_to_index["elbow_pitch_l"],
            ],
            dtype=torch.long,
            device=self.device,
        )
        self.collision_debug_enabled = bool(getattr(self.cfg.env, "collision_debug", False))
        self.collision_debug_interval = max(
            1, int(getattr(self.cfg.env, "collision_debug_interval", 200))
        )
        self.collision_debug_top_k = max(
            1, int(getattr(self.cfg.env, "collision_debug_top_k", 4))
        )
        num_penalized_bodies = len(self.penalised_contact_indices)
        self.collision_debug_contact_counts = torch.zeros(
            num_penalized_bodies,
            dtype=torch.long,
            device=self.device,
            requires_grad=False,
        )
        self.collision_debug_peak_forces = torch.zeros(
            num_penalized_bodies,
            dtype=torch.float,
            device=self.device,
            requires_grad=False,
        )
        self.standing_height_target = float(self.cfg.rewards.base_height_target)
        self.jump_height_gain_target = max(
            float(self.cfg.rewards.jump_height_target) - self.standing_height_target,
            1e-3,
        )
        self.prev_jump_contacts = torch.zeros(
            self.num_envs,
            self.feet_num,
            dtype=torch.bool,
            device=self.device,
            requires_grad=False,
        )
        self.feet_contact = torch.zeros_like(self.prev_jump_contacts)
        self.any_foot_contact = torch.zeros(
            self.num_envs,
            dtype=torch.bool,
            device=self.device,
            requires_grad=False,
        )
        self.all_foot_contact = torch.zeros(
            self.num_envs,
            dtype=torch.bool,
            device=self.device,
            requires_grad=False,
        )
        self.airborne = torch.zeros(
            self.num_envs,
            dtype=torch.bool,
            device=self.device,
            requires_grad=False,
        )
        self.airborne_time = torch.zeros(
            self.num_envs,
            dtype=torch.float,
            device=self.device,
            requires_grad=False,
        )
        self.jump_peak_height = torch.full(
            (self.num_envs,),
            self.standing_height_target,
            dtype=torch.float,
            device=self.device,
            requires_grad=False,
        )
        self.last_jump_peak_height = torch.zeros(
            self.num_envs,
            dtype=torch.float,
            device=self.device,
            requires_grad=False,
        )
        self.last_jump_air_time = torch.zeros(
            self.num_envs,
            dtype=torch.float,
            device=self.device,
            requires_grad=False,
        )
        self.valid_jump = torch.zeros(
            self.num_envs,
            dtype=torch.bool,
            device=self.device,
            requires_grad=False,
        )
        self._log_collision_body_matches()

    def reset_idx(self, env_ids):
        if len(env_ids) == 0:
            return

        super().reset_idx(env_ids)
        self.prev_jump_contacts[env_ids] = False
        self.feet_contact[env_ids] = False
        self.any_foot_contact[env_ids] = False
        self.all_foot_contact[env_ids] = False
        self.airborne[env_ids] = False
        self.airborne_time[env_ids] = 0.0
        self.jump_peak_height[env_ids] = self.standing_height_target
        self.last_jump_peak_height[env_ids] = 0.0
        self.last_jump_air_time[env_ids] = 0.0
        self.valid_jump[env_ids] = False

    def _log_collision_body_matches(self):
        print(
            f"[BruceRobot] foot_name='{self.cfg.asset.foot_name}' -> "
            f"{self._format_body_names(self.feet_body_names)}"
        )
        print(
            f"[BruceRobot] penalize_contacts_on={self.cfg.asset.penalize_contacts_on} -> "
            f"{self._format_body_names(self.penalized_contact_body_names)}"
        )
        print(
            "[BruceRobot] terminate_after_contacts_on="
            f"{self.cfg.asset.terminate_after_contacts_on} -> "
            f"{self._format_body_names(self.termination_contact_body_names)}"
        )

    def _update_collision_debug(self):
        if not self.collision_debug_enabled or len(self.penalised_contact_indices) == 0:
            return

        penalized_contact_forces = torch.norm(
            self.contact_forces[:, self.penalised_contact_indices, :], dim=-1
        )
        penalized_contacts = penalized_contact_forces > 0.1
        self.collision_debug_contact_counts += penalized_contacts.sum(dim=0).to(torch.long)
        self.collision_debug_peak_forces = torch.maximum(
            self.collision_debug_peak_forces,
            penalized_contact_forces.max(dim=0).values,
        )

        if self.common_step_counter % self.collision_debug_interval != 0:
            return

        total_hits = int(self.collision_debug_contact_counts.sum().item())
        if total_hits > 0:
            ranked_body_ids = torch.argsort(
                self.collision_debug_contact_counts, descending=True
            )
            debug_lines = []
            for body_id in ranked_body_ids[: self.collision_debug_top_k].tolist():
                body_hits = int(self.collision_debug_contact_counts[body_id].item())
                if body_hits == 0:
                    break
                peak_force = float(self.collision_debug_peak_forces[body_id].item())
                debug_lines.append(
                    f"{self.penalized_contact_body_names[body_id]}: "
                    f"hits={body_hits}, peak_force={peak_force:.2f}"
                )
            if debug_lines:
                print(
                    f"[BruceRobot][collision_debug step={self.common_step_counter}] "
                    + "; ".join(debug_lines)
                )

        self.collision_debug_contact_counts.zero_()
        self.collision_debug_peak_forces.zero_()

    def update_feet_state(self):
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        self.feet_state = self.rigid_body_states_view[:, self.feet_indices, :]
        self.feet_pos = self.feet_state[:, :, :3]
        self.feet_vel = self.feet_state[:, :, 7:10]

    def _resample_commands(self, env_ids):
        if len(env_ids) == 0:
            return
        self.commands[env_ids] = 0.0

    def _update_jump_state(self):
        raw_contact = torch.norm(self.contact_forces[:, self.feet_indices, :3], dim=2) > 1.0
        filtered_contact = torch.logical_or(raw_contact, self.prev_jump_contacts)
        self.prev_jump_contacts = raw_contact

        was_airborne = self.airborne.clone()
        self.feet_contact = filtered_contact
        self.any_foot_contact = torch.any(filtered_contact, dim=1)
        self.all_foot_contact = torch.all(filtered_contact, dim=1)
        self.airborne = ~self.any_foot_contact

        base_height = self.root_states[:, 2]
        self.valid_jump.zero_()
        self.last_jump_peak_height.zero_()
        self.last_jump_air_time.zero_()

        takeoff = self.airborne & ~was_airborne
        if torch.any(takeoff):
            self.airborne_time[takeoff] = 0.0
            self.jump_peak_height[takeoff] = base_height[takeoff]

        airborne_ids = self.airborne.nonzero(as_tuple=False).flatten()
        if len(airborne_ids) > 0:
            self.airborne_time[airborne_ids] += self.dt
            self.jump_peak_height[airborne_ids] = torch.maximum(
                self.jump_peak_height[airborne_ids],
                base_height[airborne_ids],
            )

        landing = was_airborne & self.any_foot_contact
        if torch.any(landing):
            self.last_jump_peak_height[landing] = self.jump_peak_height[landing]
            self.last_jump_air_time[landing] = self.airborne_time[landing]
            height_gain = self.last_jump_peak_height[landing] - self.standing_height_target
            self.valid_jump[landing] = (
                self.last_jump_air_time[landing] >= self.cfg.rewards.min_jump_air_time
            ) & (height_gain >= self.cfg.rewards.min_jump_height)

        grounded = ~self.airborne
        self.airborne_time[grounded] = 0.0
        self.jump_peak_height[grounded] = base_height[grounded]

    def _jump_vertical_factor(self):
        horizontal_speed_sq = torch.sum(torch.square(self.base_lin_vel[:, :2]), dim=1)
        horizontal_drift_sq = torch.sum(
            torch.square(self.root_states[:, :2] - self.env_origins[:, :2]),
            dim=1,
        )
        foot_height_diff_sq = torch.square(self.feet_pos[:, 0, 2] - self.feet_pos[:, 1, 2])

        speed_scale = max(float(self.cfg.rewards.max_horizontal_speed), 1e-3) ** 2
        drift_scale = max(float(self.cfg.rewards.max_horizontal_displacement), 1e-3) ** 2
        foot_scale = max(float(self.cfg.rewards.max_foot_height_diff), 1e-3) ** 2

        return torch.exp(
            -horizontal_speed_sq / speed_scale
            -horizontal_drift_sq / drift_scale
            -foot_height_diff_sq / foot_scale
        )

    def _post_physics_step_callback(self):
        self.update_feet_state()
        self._update_collision_debug()
        self._update_jump_state()

        return super()._post_physics_step_callback()

    def compute_observations(self):
        foot_contact = self.feet_contact.float()
        actor_obs = torch.cat(
            (
                self.base_lin_vel * self.obs_scales.lin_vel,
                self.base_ang_vel * self.obs_scales.ang_vel,
                self.projected_gravity,
                (self.dof_pos - self.default_dof_pos) * self.obs_scales.dof_pos,
                self.dof_vel * self.obs_scales.dof_vel,
                self.actions,
                foot_contact,
            ),
            dim=-1,
        )
        self.obs_buf = actor_obs
        self.privileged_obs_buf = torch.cat(
            (
                actor_obs,
                self.feet_pos[:, :, 2],
                self.root_states[:, 2:3],
            ),
            dim=-1,
        )
        if self.add_noise:
            self.obs_buf += (2 * torch.rand_like(self.obs_buf) - 1) * self.noise_scale_vec

    def _reward_jump_takeoff(self):
        takeoff_target = max(float(self.cfg.rewards.takeoff_velocity_target), 1e-3)
        upward_speed = torch.clamp(self.base_lin_vel[:, 2], min=0.0)
        return (
            self.all_foot_contact.float()
            * torch.clamp(upward_speed / takeoff_target, max=1.0)
            * self._jump_vertical_factor()
        )

    def _reward_jump_air(self):
        height_gain = torch.clamp(
            self.root_states[:, 2] - self.standing_height_target,
            min=0.0,
        )
        return (
            self.airborne.float()
            * torch.clamp(height_gain / self.jump_height_gain_target, max=1.5)
            * self._jump_vertical_factor()
        )

    def _reward_jump_height(self):
        height_gain = torch.clamp(
            self.last_jump_peak_height - self.standing_height_target,
            min=0.0,
        )
        return (
            self.valid_jump.float()
            * torch.clamp(height_gain / self.jump_height_gain_target, max=1.5)
            * self._jump_vertical_factor()
        )

    def _reward_horizontal_drift(self):
        return torch.sum(torch.square(self.base_lin_vel[:, :2]), dim=1)

    def _reward_horizontal_position(self):
        return torch.sum(
            torch.square(self.root_states[:, :2] - self.env_origins[:, :2]),
            dim=1,
        )

    def _reward_feet_symmetry(self):
        return torch.square(self.feet_pos[:, 0, 2] - self.feet_pos[:, 1, 2])

    def _reward_contact_balance(self):
        return torch.logical_xor(
            self.feet_contact[:, 0],
            self.feet_contact[:, 1],
        ).float()

    def _reward_yaw_rate(self):
        return torch.square(self.base_ang_vel[:, 2])

    def _reward_alive(self):
        return 1.0

    def _reward_contact_no_vel(self):
        contact = torch.norm(self.contact_forces[:, self.feet_indices, :3], dim=2) > 1.0
        contact_feet_vel = self.feet_vel * contact.unsqueeze(-1)
        penalize = torch.square(contact_feet_vel[:, :, :3])
        return torch.sum(penalize, dim=(1, 2))

    def _reward_hip_pos(self):
        hip_error = self.dof_pos[:, self.hip_indices] - self.default_dof_pos[:, self.hip_indices]
        return torch.sum(torch.square(hip_error), dim=1)

    def _reward_arm_pos(self):
        arm_error = self.dof_pos[:, self.arm_indices] - self.default_dof_pos[:, self.arm_indices]
        return torch.sum(torch.square(arm_error), dim=1)
