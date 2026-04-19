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

    for _ in range(horizon_steps):
        _, _, _, dones, _ = env.step(zero_actions)
        ages += 1
        done_mask = dones.bool()
        newly_done = done_mask & (first_done_step < 0)
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

    if env.viewer is not None:
        env.gym.destroy_viewer(env.viewer)
    env.gym.destroy_sim(env.sim)


if __name__ == "__main__":
    main(get_args())
