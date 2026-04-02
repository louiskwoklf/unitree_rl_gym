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
        # 3 + 3 + 3 + 16 + 16 + 16 + 2
        num_observations = 59
        # 3 + 3 + 3 + 3 + 16 + 16 + 16 + 2
        num_privileged_obs = 62
        num_actions = 16

    class commands(LeggedRobotCfg.commands):
        curriculum = True
        max_curriculum = 0.8
        heading_command = True

        class ranges(LeggedRobotCfg.commands.ranges):
            # Start with mostly forward walking and small turn/lateral demands.
            # Bruce can expand lin_vel_x through the built-in curriculum once
            # it is tracking reliably.
            lin_vel_x = [0.0, 0.6]
            lin_vel_y = [-0.1, 0.1]
            ang_vel_yaw = [-0.5, 0.5]
            heading = [-0.5, 0.5]

    class domain_rand(LeggedRobotCfg.domain_rand):
        randomize_friction = True
        friction_range = [0.5, 1.25]
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
        action_scale = 0.25
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
        max_contact_force = 60.0
        only_positive_rewards = True

        class scales(LeggedRobotCfg.rewards.scales):
            tracking_lin_vel = 1.0
            tracking_ang_vel = 0.5
            lin_vel_z = -2.0
            ang_vel_xy = -0.05
            orientation = -1.0
            base_height = -10.0
            dof_acc = 0.0#-2.5e-7
            dof_vel = 0.0#-1e-3
            feet_air_time = 0.0
            collision = -0.1#-0.5
            action_rate = -0.005#-0.01
            dof_pos_limits = -5.0
            alive = 0.15
            hip_pos = -0.5#-1.0
            arm_pos = -0.05#-0.15
            contact_no_vel = -0.2
            feet_swing_height = -20.0
            contact = 0.18


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
