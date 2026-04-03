from legged_gym.envs.base.legged_robot_config import LeggedRobotCfg, LeggedRobotCfgPPO


class BruceRoughCfg(LeggedRobotCfg):
    class init_state(LeggedRobotCfg.init_state):
        pos = [0.0, 0.0, 0.43]  # x,y,z [m]
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
        # 3 + 3 + 3 + 16 + 16 + 10 + 2
        num_observations = 53
        # actor obs + 2 foot heights + base height
        num_privileged_obs = 56
        num_actions = 10
        episode_length_s = 8.0
        collision_debug = False
        collision_debug_interval = 200
        collision_debug_top_k = 4

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
        randomize_friction = True
        friction_range = [0.8, 1.25]
        randomize_base_mass = False
        added_mass_range = [0.0, 0.0]
        push_robots = False
        push_interval_s = 5
        max_push_vel_xy = 0.8

    class control(LeggedRobotCfg.control):
        control_type = "P"
        stiffness = {
            "hip_yaw": 40.0,
            "hip_pitch": 50.0,
            "hip_roll": 50.0,
            "knee": 60.0,
            "ankle": 20.0,
            "shoulder_pitch": 20.0,
            "shoulder_roll": 20.0,
            "elbow": 12.0,
        }  # [N*m/rad]
        damping = {
            "hip_yaw": 1.0,
            "hip_pitch": 1.5,
            "hip_roll": 1.5,
            "knee": 2.0,
            "ankle": 0.8,
            "shoulder_pitch": 0.6,
            "shoulder_roll": 0.6,
            "elbow": 0.4,
        }  # [N*m*s/rad]
        # Walking used a narrow action range around the nominal stance.
        # Jumping needs enough travel to crouch and then extend aggressively.
        action_scale = 0.6
        decimation = 4

    class asset(LeggedRobotCfg.asset):
        file = "{LEGGED_GYM_ROOT_DIR}/resources/robots/bruce/bruce.urdf"
        name = "bruce"
        # Bruce's toe/heel spheres are collision shapes inside the ankle links,
        # so the foot bodies exposed to Isaac Gym are the ankle links.
        foot_name = "ankle_pitch_link"
        penalize_contacts_on = ["hip", "knee"]#, "shoulder", "elbow"]
        terminate_after_contacts_on = ["base_link"]
        self_collisions = 0  # 1 to disable, 0 to enable...bitwise filter
        flip_visual_attachments = False

    class rewards(LeggedRobotCfg.rewards):
        soft_dof_pos_limit = 0.9
        base_height_target = 0.43
        jump_height_target = 0.55
        takeoff_velocity_target = 0.9
        min_jump_air_time = 0.06
        min_jump_height = 0.05
        max_horizontal_speed = 0.15
        max_horizontal_displacement = 0.06
        max_foot_height_diff = 0.03
        max_pitch_angle = 0.2
        max_pitch_rate = 1.5
        landing_reward_window_s = 0.18
        max_contact_force = 60.0
        only_positive_rewards = False

        class scales(LeggedRobotCfg.rewards.scales):
            termination = -5.0
            tracking_lin_vel = 0.0
            tracking_ang_vel = 0.0
            lin_vel_z = 0.0
            ang_vel_xy = -0.1
            orientation = -2.0
            base_height = 0.0
            dof_acc = 0.0
            dof_vel = 0.0
            feet_air_time = 0.0
            collision = -0.2
            action_rate = -0.01
            dof_pos_limits = -5.0
            alive = 0.0
            hip_pos = -0.4
            arm_pos = -0.05
            contact_no_vel = 0.0
            feet_swing_height = 0.0
            contact = 0.0
            jump_takeoff = 1.5
            jump_air = 2.0
            jump_height = 4.0
            horizontal_drift = -6.0
            horizontal_position = -8.0
            feet_symmetry = -4.0
            contact_balance = -2.0
            pitch_angle = -6.0
            pitch_rate = -2.5
            landing_upright = 6.0
            yaw_rate = -0.3


class BruceRoughCfgPPO(LeggedRobotCfgPPO):
    class policy:
        init_noise_std = 0.8
        actor_hidden_dims = [32]
        critic_hidden_dims = [32]
        activation = "elu"
        rnn_type = "lstm"
        rnn_hidden_size = 64
        rnn_num_layers = 1

    class algorithm(LeggedRobotCfgPPO.algorithm):
        entropy_coef = 0.01

    class runner(LeggedRobotCfgPPO.runner):
        policy_class_name = "ActorCriticRecurrent"
        max_iterations = 10000
        run_name = ""
        experiment_name = "bruce"
