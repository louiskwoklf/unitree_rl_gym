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


def _install_reset_snapshot_hook(env, ankle_index_tensor):
    snapshot = {
        "valid": torch.zeros(env.num_envs, dtype=torch.bool, device=env.device),
        "base_height": torch.full((env.num_envs,), float("nan"), dtype=torch.float, device=env.device),
        "roll": torch.full((env.num_envs,), float("nan"), dtype=torch.float, device=env.device),
        "pitch": torch.full((env.num_envs,), float("nan"), dtype=torch.float, device=env.device),
        "timeout": torch.zeros(env.num_envs, dtype=torch.bool, device=env.device),
        "tip": torch.zeros(env.num_envs, dtype=torch.bool, device=env.device),
        "contact": torch.zeros(env.num_envs, dtype=torch.bool, device=env.device),
    }
    if ankle_index_tensor is not None:
        snapshot["ankle_pos"] = torch.full(
            (env.num_envs, len(ankle_index_tensor)),
            float("nan"),
            dtype=torch.float,
            device=env.device,
        )
        snapshot["ankle_vel"] = torch.full_like(snapshot["ankle_pos"], float("nan"))

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
            if ankle_index_tensor is not None:
                snapshot["ankle_pos"][env_ids] = env.dof_pos[env_ids][:, ankle_index_tensor]
                snapshot["ankle_vel"][env_ids] = env.dof_vel[env_ids][:, ankle_index_tensor]
        return original_reset_idx(env_ids)

    env.reset_idx = wrapped_reset_idx
    return snapshot


def main(args):
    env_cfg, _ = task_registry.get_cfgs(name=args.task)
    _disable_task_motion(env_cfg)

    if args.num_envs is None:
        env_cfg.env.num_envs = 1 if not args.headless else min(env_cfg.env.num_envs, 256)

    env, _ = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)
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

    ankle_dofs = _find_dof_indices(env, ("ankle_pitch_r", "ankle_pitch_l"))
    ankle_names = [name for name, _ in ankle_dofs]
    ankle_indices = [idx for _, idx in ankle_dofs]
    if ankle_indices:
        ankle_index_tensor = torch.tensor(ankle_indices, dtype=torch.long, device=env.device)
        ankle_pos_error_max = torch.zeros(
            env.num_envs,
            len(ankle_indices),
            dtype=torch.float,
            device=env.device,
        )
        ankle_vel_abs_max = torch.zeros_like(ankle_pos_error_max)
        first_done_ankle_pos = torch.full_like(ankle_pos_error_max, float("nan"))
        first_done_ankle_vel = torch.full_like(ankle_pos_error_max, float("nan"))
    else:
        ankle_index_tensor = None
    reset_snapshot = _install_reset_snapshot_hook(env, ankle_index_tensor)

    for _ in range(horizon_steps):
        _, _, _, dones, _ = env.step(zero_actions)
        ages += 1
        done_mask = dones.bool()
        newly_done = done_mask & (first_done_step < 0)

        if ankle_index_tensor is not None:
            ankle_pos_error = torch.abs(
                env.dof_pos[:, ankle_index_tensor] - env.default_dof_pos[:, ankle_index_tensor]
            )
            ankle_vel_abs = torch.abs(env.dof_vel[:, ankle_index_tensor])
            ankle_pos_error_max = torch.maximum(ankle_pos_error_max, ankle_pos_error)
            ankle_vel_abs_max = torch.maximum(ankle_vel_abs_max, ankle_vel_abs)

        if newly_done.any():
            first_done_base_height[newly_done] = reset_snapshot["base_height"][newly_done]
            first_done_roll[newly_done] = reset_snapshot["roll"][newly_done]
            first_done_pitch[newly_done] = reset_snapshot["pitch"][newly_done]
            first_done_timeout[newly_done] = reset_snapshot["timeout"][newly_done]
            first_done_tip[newly_done] = reset_snapshot["tip"][newly_done]
            first_done_contact[newly_done] = reset_snapshot["contact"][newly_done]
            if ankle_index_tensor is not None:
                first_done_ankle_pos[newly_done] = reset_snapshot["ankle_pos"][newly_done]
                first_done_ankle_vel[newly_done] = reset_snapshot["ankle_vel"][newly_done]
            reset_snapshot["valid"][newly_done] = False

        first_done_step[newly_done] = ages[newly_done]
        ages[done_mask] = 0

    survived_mask = first_done_step < 0
    failed_mask = ~survived_mask

    print("=" * 80)
    print(f"Zero-action stability check for task '{args.task}'")
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
        if ankle_index_tensor is not None:
            pos_mean = first_done_ankle_pos[failed_mask].mean(dim=0).detach().cpu()
            vel_mean = first_done_ankle_vel[failed_mask].mean(dim=0).detach().cpu()
            for i, ankle_name in enumerate(ankle_names):
                print(
                    f"First-failure {ankle_name}: "
                    f"pos_mean={pos_mean[i].item():.4f} rad, "
                    f"vel_mean={vel_mean[i].item():.4f} rad/s"
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

    if ankle_index_tensor is not None:
        ankle_pos_error_max_cpu = ankle_pos_error_max.detach().cpu()
        ankle_vel_abs_max_cpu = ankle_vel_abs_max.detach().cpu()
        for i, ankle_name in enumerate(ankle_names):
            print(
                f"Rollout {ankle_name}: "
                f"|pos_error|_max_mean={ankle_pos_error_max_cpu[:, i].mean().item():.4f} rad, "
                f"|vel|_max_mean={ankle_vel_abs_max_cpu[:, i].mean().item():.4f} rad/s"
            )

    if env.viewer is not None:
        env.gym.destroy_viewer(env.viewer)
    env.gym.destroy_sim(env.sim)


if __name__ == "__main__":
    main(get_args())
