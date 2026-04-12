from legged_gym.envs.base.legged_robot_config import LeggedRobotCfg, LeggedRobotCfgPPO


class BruceStandCfg(LeggedRobotCfg):
    class init_state(LeggedRobotCfg.init_state):
        pos = [0.0, 0.0, 0.43]
        default_joint_angles = {
            "hip_yaw_r": -0.008243,
            "hip_pitch_r": 0.469268,
            "hip_roll_r": 0.018225,
            "knee_pitch_r": -0.947148,
            "ankle_pitch_r": 0.477813,
            "hip_yaw_l": 0.008243,
            "hip_pitch_l": 0.469268,
            "hip_roll_l": -0.018225,
            "knee_pitch_l": -0.947148,
            "ankle_pitch_l": 0.477813,
            "shoulder_pitch_r": -0.7,
            "shoulder_roll_r": 1.3,
            "elbow_pitch_r": 2.0,
            "shoulder_pitch_l": 0.7,
            "shoulder_roll_l": -1.3,
            "elbow_pitch_l": -2.0,
        }

    class env(LeggedRobotCfg.env):
        num_observations = 54
        num_privileged_obs = None
        num_actions = 10
        episode_length_s = 8.0

    class commands(LeggedRobotCfg.commands):
        curriculum = False
        max_curriculum = 0.0
        heading_command = False

        class ranges(LeggedRobotCfg.commands.ranges):
            lin_vel_x = [0.0, 0.0]
            lin_vel_y = [0.0, 0.0]
            ang_vel_yaw = [0.0, 0.0]
            heading = [0.0, 0.0]

    class domain_rand(LeggedRobotCfg.domain_rand):
        randomize_friction = False
        randomize_base_mass = False
        added_mass_range = [0.0, 0.0]
        push_robots = False

    class control(LeggedRobotCfg.control):
        control_type = "P"
        stiffness = {
            "hip_yaw": 40.0,
            "hip_pitch": 50.0,
            "hip_roll": 50.0,
            "knee": 60.0,
            "ankle": 25.0,
            "shoulder_pitch": 20.0,
            "shoulder_roll": 20.0,
            "elbow": 12.0,
        }
        damping = {
            "hip_yaw": 1.0,
            "hip_pitch": 1.5,
            "hip_roll": 1.5,
            "knee": 2.0,
            "ankle": 1.0,
            "shoulder_pitch": 0.6,
            "shoulder_roll": 0.6,
            "elbow": 0.4,
        }
        action_scale = 0.2
        decimation = 4

    class asset(LeggedRobotCfg.asset):
        file = "{LEGGED_GYM_ROOT_DIR}/resources/robots/bruce/bruce.urdf"
        name = "bruce"
        foot_name = "ankle_pitch_link"
        penalize_contacts_on = ["hip", "knee", "shoulder", "elbow"]
        terminate_after_contacts_on = ["base_link"]
        self_collisions = 0
        flip_visual_attachments = False

    class rewards(LeggedRobotCfg.rewards):
        only_positive_rewards = True
        soft_dof_pos_limit = 0.9
        base_height_target = 0.43

        class scales:
            termination = -5.0
            alive = 0.3
            orientation = -5.0
            ang_vel_xy = -0.2
            lin_vel_z = -2.0
            base_height = -10.0
            stand_still = -1.0
            dof_vel = -1.0e-3
            dof_acc = -2.5e-7
            torques = -1.0e-5
            action_rate = -0.01
            collision = -1.0
            dof_pos_limits = -5.0

    class noise(LeggedRobotCfg.noise):
        add_noise = False


class BruceStandCfgPPO(LeggedRobotCfgPPO):
    class policy(LeggedRobotCfgPPO.policy):
        init_noise_std = 0.3
        actor_hidden_dims = [128, 64]
        critic_hidden_dims = [128, 64]
        activation = "elu"

    class algorithm(LeggedRobotCfgPPO.algorithm):
        entropy_coef = 0.005

    class runner(LeggedRobotCfgPPO.runner):
        max_iterations = 3000
        run_name = ""
        experiment_name = "bruce_stand"
