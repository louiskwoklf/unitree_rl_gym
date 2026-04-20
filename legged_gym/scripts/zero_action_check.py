import isaacgym

import torch

from legged_gym.envs import *
from legged_gym.utils import get_args, task_registry


def _disable_task_motion(env_cfg):
    if hasattr(env_cfg, "commands"):
        env_cfg.commands.curriculum = False
        env_cfg.commands.max_curriculum = 0.0
        env_cfg.commands.heading_command = False
        for name in ("lin_vel_x", "lin_vel_y", "ang_vel_yaw", "heading"):
            if hasattr(env_cfg.commands.ranges, name):
                setattr(env_cfg.commands.ranges, name, [0.0, 0.0])

    if hasattr(env_cfg, "noise"):
        env_cfg.noise.add_noise = False

    if hasattr(env_cfg, "domain_rand"):
        env_cfg.domain_rand.randomize_friction = False
        env_cfg.domain_rand.randomize_base_mass = False
        env_cfg.domain_rand.added_mass_range = [0.0, 0.0]
        env_cfg.domain_rand.push_robots = False


def _find_dof_indices(env, names):
    name_to_index = {name: i for i, name in enumerate(getattr(env, "dof_names", []))}
    return [(name, name_to_index[name]) for name in names if name in name_to_index]


def _select_diagnostic_dofs(env):
    preferred_names = (
        "hip_pitch_r",
        "knee_pitch_r",
        "ankle_pitch_r",
        "hip_pitch_l",
        "knee_pitch_l",
        "ankle_pitch_l",
    )
    selected = _find_dof_indices(env, preferred_names)
    if selected:
        return selected
    if hasattr(env, "leg_indices"):
        leg_indices = env.leg_indices.detach().cpu().tolist()
        return [(env.dof_names[idx], idx) for idx in leg_indices]
    return _find_dof_indices(env, ("ankle_pitch_r", "ankle_pitch_l"))


def _get_action_joint_names(env):
    if hasattr(env, "LEG_JOINT_NAMES"):
        return list(env.LEG_JOINT_NAMES)
    if hasattr(env, "leg_indices") and env.num_actions == len(env.leg_indices):
        leg_indices = env.leg_indices.detach().cpu().tolist()
        return [env.dof_names[idx] for idx in leg_indices]
    if env.num_actions == len(getattr(env, "dof_names", [])):
        return list(env.dof_names)
    return None


def _install_reset_snapshot_hook(env, joint_index_tensor):
    snapshot = {
        "valid": torch.zeros(env.num_envs, dtype=torch.bool, device=env.device),
        "base_height": torch.full((env.num_envs,), float("nan"), dtype=torch.float, device=env.device),
        "roll": torch.full((env.num_envs,), float("nan"), dtype=torch.float, device=env.device),
        "pitch": torch.full((env.num_envs,), float("nan"), dtype=torch.float, device=env.device),
        "timeout": torch.zeros(env.num_envs, dtype=torch.bool, device=env.device),
        "tip": torch.zeros(env.num_envs, dtype=torch.bool, device=env.device),
        "contact": torch.zeros(env.num_envs, dtype=torch.bool, device=env.device),
    }
    if joint_index_tensor is not None:
        snapshot["joint_pos"] = torch.full(
            (env.num_envs, len(joint_index_tensor)),
            float("nan"),
            dtype=torch.float,
            device=env.device,
        )
        snapshot["joint_vel"] = torch.full_like(snapshot["joint_pos"], float("nan"))
        snapshot["joint_torque"] = torch.full_like(snapshot["joint_pos"], float("nan"))
        snapshot["joint_limit_margin"] = torch.full_like(snapshot["joint_pos"], float("nan"))
        snapshot["joint_torque_ratio"] = torch.full_like(snapshot["joint_pos"], float("nan"))

    original_reset_idx = env.reset_idx

    def wrapped_reset_idx(env_ids):
        if len(env_ids) > 0:
            snapshot["valid"][env_ids] = True
            snapshot["base_height"][env_ids] = env.root_states[env_ids, 2]
            snapshot["roll"][env_ids] = torch.abs(env.rpy[env_ids, 0])
            snapshot["pitch"][env_ids] = torch.abs(env.rpy[env_ids, 1])
            snapshot["timeout"][env_ids] = env.time_out_buf[env_ids]
            snapshot["tip"][env_ids] = torch.logical_or(
                torch.abs(env.rpy[env_ids, 1]) > 1.0,
                torch.abs(env.rpy[env_ids, 0]) > 0.8,
            )
            if len(env.termination_contact_indices) > 0:
                termination_contact = torch.any(
                    torch.norm(
                        env.contact_forces[:, env.termination_contact_indices, :], dim=-1
                    )
                    > 1.0,
                    dim=1,
                )
                snapshot["contact"][env_ids] = termination_contact[env_ids]
            if joint_index_tensor is not None:
                snapshot["joint_pos"][env_ids] = env.dof_pos[env_ids][:, joint_index_tensor]
                snapshot["joint_vel"][env_ids] = env.dof_vel[env_ids][:, joint_index_tensor]
                snapshot["joint_torque"][env_ids] = env.torques[env_ids][:, joint_index_tensor]
                lower_margin = env.dof_pos[env_ids][:, joint_index_tensor] - env.dof_pos_limits[joint_index_tensor, 0].unsqueeze(0)
                upper_margin = env.dof_pos_limits[joint_index_tensor, 1].unsqueeze(0) - env.dof_pos[env_ids][:, joint_index_tensor]
                snapshot["joint_limit_margin"][env_ids] = torch.minimum(lower_margin, upper_margin)
                snapshot["joint_torque_ratio"][env_ids] = torch.abs(
                    env.torques[env_ids][:, joint_index_tensor]
                ) / torch.clamp(env.torque_limits[joint_index_tensor].unsqueeze(0), min=1e-6)
        return original_reset_idx(env_ids)

    env.reset_idx = wrapped_reset_idx
    return snapshot


def _override_effort_limit(env, effort_limit):
    original_min = float(env.torque_limits.min().item())
    original_max = float(env.torque_limits.max().item())

    if hasattr(env.gym, "get_actor_dof_properties"):
        for env_handle, actor_handle in zip(env.envs, env.actor_handles):
            dof_props = env.gym.get_actor_dof_properties(env_handle, actor_handle)
            dof_props["effort"][:] = effort_limit
            env.gym.set_actor_dof_properties(env_handle, actor_handle, dof_props)

    env.torque_limits[:] = effort_limit
    return original_min, original_max


def _print_mode_header(title, env, args, torque_override):
    print("=" * 80)
    print(f"{title} for task '{args.task}'")
    print(
        "Mode: "
        f"fix_base_link={bool(getattr(env.cfg.asset, 'fix_base_link', False))}, "
        f"headless={bool(args.headless)}, "
        f"override_effort_limit={args.override_effort_limit if args.override_effort_limit is not None else 'None'}"
    )
    if torque_override is not None:
        original_min, original_max = torque_override
        print(
            "Effort limit override: "
            f"original_min={original_min:.4f} Nm, "
            f"original_max={original_max:.4f} Nm, "
            f"new_limit={float(args.override_effort_limit):.4f} Nm"
        )


def _run_joint_step_response(env, args, torque_override):
    action_joint_names = _get_action_joint_names(env)
    if action_joint_names is None:
        raise ValueError(
            "This environment does not expose a direct mapping from action indices to joint names."
        )
    if args.step_joint not in action_joint_names:
        raise ValueError(
            f"Unknown step joint '{args.step_joint}'. "
            f"Available action joints: {', '.join(action_joint_names)}"
        )

    dof_matches = _find_dof_indices(env, (args.step_joint,))
    if not dof_matches:
        raise ValueError(f"Joint '{args.step_joint}' is not present in env.dof_names")

    dof_index = dof_matches[0][1]
    action_index = action_joint_names.index(args.step_joint)
    warmup_steps = max(0, int(args.step_warmup_steps))
    step_steps = max(1, int(args.step_steps))
    action_magnitude = float(args.step_action)
    action_scale = float(env.cfg.control.action_scale)
    default_pos = float(env.default_dof_pos[0, dof_index].item())
    dof_lower = float(env.dof_pos_limits[dof_index, 0].item())
    dof_upper = float(env.dof_pos_limits[dof_index, 1].item())
    zero_actions = torch.zeros(env.num_envs, env.num_actions, device=env.device, requires_grad=False)

    _print_mode_header("Joint step-response check", env, args, torque_override)
    print(
        f"Joint: {args.step_joint} "
        f"(action_index={action_index}, dof_index={dof_index})"
    )
    print(
        "Step parameters: "
        f"action_magnitude={action_magnitude:.4f}, "
        f"target_shift={action_scale * action_magnitude:.4f} rad, "
        f"warmup_steps={warmup_steps}, "
        f"response_steps={step_steps} ({step_steps * env.dt:.2f}s)"
    )
    print(
        "Joint limits: "
        f"soft_lower={dof_lower:.4f} rad, "
        f"soft_upper={dof_upper:.4f} rad, "
        f"default={default_pos:.4f} rad"
    )

    for phase_name, action_value in (("positive", action_magnitude), ("negative", -action_magnitude)):
        env.reset()
        done_step = None
        for warmup_idx in range(warmup_steps):
            _, _, _, dones, _ = env.step(zero_actions)
            if bool(dones[0].item()):
                done_step = -(warmup_idx + 1)
                break

        baseline_pos = float(env.dof_pos[0, dof_index].item())
        baseline_vel = float(env.dof_vel[0, dof_index].item())
        target_pos = default_pos + action_scale * action_value
        delta_min = 0.0
        delta_max = 0.0
        vel_abs_max = 0.0
        torque_abs_max = 0.0
        torque_ratio_max = 0.0
        limit_margin_min = float("inf")

        actions = zero_actions.clone()
        actions[:, action_index] = action_value
        if done_step is None:
            for step_idx in range(step_steps):
                _, _, _, dones, _ = env.step(actions)
                pos = float(env.dof_pos[0, dof_index].item())
                vel = float(env.dof_vel[0, dof_index].item())
                torque = float(env.torques[0, dof_index].item())
                torque_ratio = abs(torque) / max(float(env.torque_limits[dof_index].item()), 1e-6)
                lower_margin = pos - float(env.dof_pos_limits[dof_index, 0].item())
                upper_margin = float(env.dof_pos_limits[dof_index, 1].item()) - pos
                limit_margin = min(lower_margin, upper_margin)
                delta = pos - baseline_pos
                delta_min = min(delta_min, delta)
                delta_max = max(delta_max, delta)
                vel_abs_max = max(vel_abs_max, abs(vel))
                torque_abs_max = max(torque_abs_max, abs(torque))
                torque_ratio_max = max(torque_ratio_max, torque_ratio)
                limit_margin_min = min(limit_margin_min, limit_margin)
                if bool(dones[0].item()):
                    done_step = step_idx + 1
                    break

        final_pos = float(env.dof_pos[0, dof_index].item())
        final_vel = float(env.dof_vel[0, dof_index].item())
        final_delta = final_pos - baseline_pos
        expected_peak = delta_max if action_value > 0.0 else -delta_min
        opposite_peak = -delta_min if action_value > 0.0 else delta_max
        direction_match = expected_peak > opposite_peak
        done_desc = "none" if done_step is None else (
            f"warmup_{abs(done_step)}" if done_step < 0 else f"response_{done_step}"
        )

        print(
            f"Phase {phase_name} ({action_value:+.4f}): "
            f"target={target_pos:.4f} rad, "
            f"baseline={baseline_pos:.4f} rad, "
            f"final={final_pos:.4f} rad, "
            f"final_delta={final_delta:+.4f} rad, "
            f"delta_range=[{delta_min:+.4f}, {delta_max:+.4f}] rad, "
            f"final_vel={final_vel:+.4f} rad/s, "
            f"|vel|_max={vel_abs_max:.4f} rad/s, "
            f"|torque|_max={torque_abs_max:.4f} Nm, "
            f"torque_ratio_max={torque_ratio_max:.3f}, "
            f"limit_margin_min={limit_margin_min:.4f} rad, "
            f"direction_match={direction_match}, "
            f"done={done_desc}"
        )


def main(args):
    env_cfg, _ = task_registry.get_cfgs(name=args.task)
    _disable_task_motion(env_cfg)

    if args.num_envs is None:
        env_cfg.env.num_envs = 1 if not args.headless else min(env_cfg.env.num_envs, 256)

    env, _ = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)
    torque_override = None
    if args.override_effort_limit is not None:
        torque_override = _override_effort_limit(env, float(args.override_effort_limit))
    if args.step_joint is not None:
        _run_joint_step_response(env, args, torque_override)
        if env.viewer is not None:
            env.gym.destroy_viewer(env.viewer)
        env.gym.destroy_sim(env.sim)
        return
    env.reset()

    horizon_steps = int(env.max_episode_length)
    zero_actions = torch.zeros(
        env.num_envs,
        env.num_actions,
        device=env.device,
        requires_grad=False,
    )

    ages = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)
    first_done_step = torch.full(
        (env.num_envs,),
        -1,
        dtype=torch.long,
        device=env.device,
    )
    first_done_base_height = torch.full(
        (env.num_envs,),
        float("nan"),
        dtype=torch.float,
        device=env.device,
    )
    first_done_roll = torch.full(
        (env.num_envs,),
        float("nan"),
        dtype=torch.float,
        device=env.device,
    )
    first_done_pitch = torch.full(
        (env.num_envs,),
        float("nan"),
        dtype=torch.float,
        device=env.device,
    )
    first_done_timeout = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    first_done_tip = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    first_done_contact = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)

    diagnostic_dofs = _select_diagnostic_dofs(env)
    diagnostic_names = [name for name, _ in diagnostic_dofs]
    diagnostic_indices = [idx for _, idx in diagnostic_dofs]
    if diagnostic_indices:
        joint_index_tensor = torch.tensor(diagnostic_indices, dtype=torch.long, device=env.device)
        joint_target = env.default_dof_pos[0, joint_index_tensor]
        joint_pos_error_max = torch.zeros(
            env.num_envs,
            len(diagnostic_indices),
            dtype=torch.float,
            device=env.device,
        )
        joint_vel_abs_max = torch.zeros_like(joint_pos_error_max)
        joint_torque_abs_max = torch.zeros_like(joint_pos_error_max)
        joint_torque_ratio_max = torch.zeros_like(joint_pos_error_max)
        joint_limit_margin_min = torch.full_like(joint_pos_error_max, float("inf"))
        first_done_joint_pos = torch.full_like(joint_pos_error_max, float("nan"))
        first_done_joint_vel = torch.full_like(joint_pos_error_max, float("nan"))
        first_done_joint_torque = torch.full_like(joint_pos_error_max, float("nan"))
        first_done_joint_limit_margin = torch.full_like(joint_pos_error_max, float("nan"))
        first_done_joint_torque_ratio = torch.full_like(joint_pos_error_max, float("nan"))
    else:
        joint_index_tensor = None
        joint_target = None
    reset_snapshot = _install_reset_snapshot_hook(env, joint_index_tensor)

    for _ in range(horizon_steps):
        _, _, _, dones, _ = env.step(zero_actions)
        ages += 1
        done_mask = dones.bool()
        newly_done = done_mask & (first_done_step < 0)

        if joint_index_tensor is not None:
            joint_pos_error = torch.abs(
                env.dof_pos[:, joint_index_tensor] - joint_target.unsqueeze(0)
            )
            joint_vel_abs = torch.abs(env.dof_vel[:, joint_index_tensor])
            joint_torque_abs = torch.abs(env.torques[:, joint_index_tensor])
            joint_torque_ratio = joint_torque_abs / torch.clamp(
                env.torque_limits[joint_index_tensor].unsqueeze(0), min=1e-6
            )
            lower_margin = env.dof_pos[:, joint_index_tensor] - env.dof_pos_limits[joint_index_tensor, 0].unsqueeze(0)
            upper_margin = env.dof_pos_limits[joint_index_tensor, 1].unsqueeze(0) - env.dof_pos[:, joint_index_tensor]
            joint_limit_margin = torch.minimum(lower_margin, upper_margin)
            joint_pos_error_max = torch.maximum(joint_pos_error_max, joint_pos_error)
            joint_vel_abs_max = torch.maximum(joint_vel_abs_max, joint_vel_abs)
            joint_torque_abs_max = torch.maximum(joint_torque_abs_max, joint_torque_abs)
            joint_torque_ratio_max = torch.maximum(joint_torque_ratio_max, joint_torque_ratio)
            joint_limit_margin_min = torch.minimum(joint_limit_margin_min, joint_limit_margin)

        if newly_done.any():
            first_done_base_height[newly_done] = reset_snapshot["base_height"][newly_done]
            first_done_roll[newly_done] = reset_snapshot["roll"][newly_done]
            first_done_pitch[newly_done] = reset_snapshot["pitch"][newly_done]
            first_done_timeout[newly_done] = reset_snapshot["timeout"][newly_done]
            first_done_tip[newly_done] = reset_snapshot["tip"][newly_done]
            first_done_contact[newly_done] = reset_snapshot["contact"][newly_done]
            if joint_index_tensor is not None:
                first_done_joint_pos[newly_done] = reset_snapshot["joint_pos"][newly_done]
                first_done_joint_vel[newly_done] = reset_snapshot["joint_vel"][newly_done]
                first_done_joint_torque[newly_done] = reset_snapshot["joint_torque"][newly_done]
                first_done_joint_limit_margin[newly_done] = reset_snapshot["joint_limit_margin"][newly_done]
                first_done_joint_torque_ratio[newly_done] = reset_snapshot["joint_torque_ratio"][newly_done]
                reset_joint_pos_error = torch.abs(
                    reset_snapshot["joint_pos"][newly_done] - joint_target.unsqueeze(0)
                )
                reset_joint_vel_abs = torch.abs(reset_snapshot["joint_vel"][newly_done])
                reset_joint_torque_abs = torch.abs(reset_snapshot["joint_torque"][newly_done])
                joint_pos_error_max[newly_done] = torch.maximum(
                    joint_pos_error_max[newly_done], reset_joint_pos_error
                )
                joint_vel_abs_max[newly_done] = torch.maximum(
                    joint_vel_abs_max[newly_done], reset_joint_vel_abs
                )
                joint_torque_abs_max[newly_done] = torch.maximum(
                    joint_torque_abs_max[newly_done], reset_joint_torque_abs
                )
                joint_torque_ratio_max[newly_done] = torch.maximum(
                    joint_torque_ratio_max[newly_done], reset_snapshot["joint_torque_ratio"][newly_done]
                )
                joint_limit_margin_min[newly_done] = torch.minimum(
                    joint_limit_margin_min[newly_done], reset_snapshot["joint_limit_margin"][newly_done]
                )
            reset_snapshot["valid"][newly_done] = False

        first_done_step[newly_done] = ages[newly_done]
        ages[done_mask] = 0

    survived_mask = first_done_step < 0
    failed_mask = ~survived_mask

    _print_mode_header("Zero-action stability check", env, args, torque_override)
    print(
        f"Evaluated {env.num_envs} envs for {horizon_steps} policy steps "
        f"({horizon_steps * env.dt:.2f}s)"
    )
    print(
        f"Full-horizon survivors: {int(survived_mask.sum().item())}/{env.num_envs}"
    )

    if failed_mask.any():
        fail_steps = first_done_step[failed_mask].float()
        print(
            "Early-failure step stats: "
            f"min={int(fail_steps.min().item())}, "
            f"mean={fail_steps.mean().item():.1f}, "
            f"max={int(fail_steps.max().item())}"
        )
        failed_mask_cpu = failed_mask.detach().cpu()
        fail_time_s = first_done_step[failed_mask].float().mean().item() * env.dt
        print(f"Mean first-failure time: {fail_time_s:.3f}s")
        print(
            "First-failure causes: "
            f"tip={int(first_done_tip[failed_mask].sum().item())}, "
            f"base_contact={int(first_done_contact[failed_mask].sum().item())}, "
            f"time_out={int(first_done_timeout[failed_mask].sum().item())}"
        )
        print(
            "First-failure posture stats: "
            f"base_height_mean={first_done_base_height[failed_mask].mean().item():.4f} m, "
            f"|roll|_mean={first_done_roll[failed_mask].mean().item():.4f} rad, "
            f"|pitch|_mean={first_done_pitch[failed_mask].mean().item():.4f} rad"
        )
        if joint_index_tensor is not None:
            pos_mean = first_done_joint_pos[failed_mask].mean(dim=0).detach().cpu()
            vel_mean = first_done_joint_vel[failed_mask].mean(dim=0).detach().cpu()
            torque_mean = first_done_joint_torque[failed_mask].mean(dim=0).detach().cpu()
            limit_margin_mean = first_done_joint_limit_margin[failed_mask].mean(dim=0).detach().cpu()
            torque_ratio_mean = first_done_joint_torque_ratio[failed_mask].mean(dim=0).detach().cpu()
            for i, joint_name in enumerate(diagnostic_names):
                print(
                    f"First-failure {joint_name}: "
                    f"pos_mean={pos_mean[i].item():.4f} rad, "
                    f"vel_mean={vel_mean[i].item():.4f} rad/s, "
                    f"torque_mean={torque_mean[i].item():.4f} Nm, "
                    f"torque_ratio_mean={torque_ratio_mean[i].item():.3f}, "
                    f"limit_margin_mean={limit_margin_mean[i].item():.4f} rad"
                )
    else:
        print("No env reset before the horizon.")

    base_height = env.root_states[:, 2].detach().cpu()
    roll = env.rpy[:, 0].abs().detach().cpu()
    pitch = env.rpy[:, 1].abs().detach().cpu()

    if survived_mask.any():
        survived_mask_cpu = survived_mask.detach().cpu()
        print(
            "Survivor posture stats: "
            f"base_height_mean={base_height[survived_mask_cpu].mean().item():.4f} m, "
            f"|roll|_mean={roll[survived_mask_cpu].mean().item():.4f} rad, "
            f"|pitch|_mean={pitch[survived_mask_cpu].mean().item():.4f} rad"
        )
    else:
        print(
            "Final posture stats: "
            f"base_height_mean={base_height.mean().item():.4f} m, "
            f"|roll|_mean={roll.mean().item():.4f} rad, "
            f"|pitch|_mean={pitch.mean().item():.4f} rad"
        )

    if joint_index_tensor is not None:
        joint_pos_error_max_cpu = joint_pos_error_max.detach().cpu()
        joint_vel_abs_max_cpu = joint_vel_abs_max.detach().cpu()
        joint_torque_abs_max_cpu = joint_torque_abs_max.detach().cpu()
        joint_torque_ratio_max_cpu = joint_torque_ratio_max.detach().cpu()
        joint_limit_margin_min_cpu = joint_limit_margin_min.detach().cpu()
        for i, joint_name in enumerate(diagnostic_names):
            print(
                f"Rollout {joint_name}: "
                f"|pos_error|_max_mean={joint_pos_error_max_cpu[:, i].mean().item():.4f} rad, "
                f"|vel|_max_mean={joint_vel_abs_max_cpu[:, i].mean().item():.4f} rad/s, "
                f"|torque|_max_mean={joint_torque_abs_max_cpu[:, i].mean().item():.4f} Nm, "
                f"torque_ratio_max_mean={joint_torque_ratio_max_cpu[:, i].mean().item():.3f}, "
                f"limit_margin_min_mean={joint_limit_margin_min_cpu[:, i].mean().item():.4f} rad"
            )

    if env.viewer is not None:
        env.gym.destroy_viewer(env.viewer)
    env.gym.destroy_sim(env.sim)


if __name__ == "__main__":
    main(get_args())
