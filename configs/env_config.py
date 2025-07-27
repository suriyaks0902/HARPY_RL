# config for test, play.py
config_play = {
    "max_timesteps": 2500,  # NOTE: max timestep per episode
    "is_visual": True,  # NOTE: set true to visualize robot
    "cam_track_robot": False,  # NOTE: set true to let camera track the robot
    "step_zerotorque": False,  # NOTE: set true to step without torque
    "minimal_rand": False,  # NOTE: set true to use minimal rand range -> only rand floor friction
    "is_noisy": True,  # NOTE: set true to add noise on observation
    "add_perturbation": False,  # NOTE: set true to add external perturbation
    "add_standing": True,  # NOTE: set true to include standing skill
    "fixed_gait": True,  # NOTE: set true to only test one cmd of 'fixed_gait_cmd'
    "add_rotation": False,  # NOTE: set true to add rotation (vyaw) command, only useful if not fixed gait
    "fixed_gait_cmd": [
        1.0,  # vx, m/s
        0.0,  # vy, m/s
        0.98,  # walking height, m
        0.0,  # turning rate, deg/s
    ],  # fixed gait cmd
}

# config for trainig, train.py
config_train = {
    "max_timesteps": 2500,  # NOTE: max timestep per episode
    "is_visual": False,  # NOTE: set true to visualize robot
    "cam_track_robot": False,  # NOTE: set true to let camera track the robot
    "step_zerotorque": False,  # NOTE: set true to step without torque
    "minimal_rand": False,  # NOTE: set true to use minimal rand range -> only rand floor friction
    "is_noisy": False,  # NOTE: set true to add noise on observation
    "add_perturbation": False,  # NOTE: set true to add external perturbation
    "add_standing": True,  # NOTE: set true to include standing skill
    "fixed_gait": True,  # NOTE: set true to only test one cmd of 'fixed_gait_cmd'
    "add_rotation": True,  # NOTE: set true to add rotation (vyaw) command, only useful if not fixed gait
    "fixed_gait_cmd": [
        0.0,
        0.0,
        0.98,
        0.0,
    ],  # fixed gait cmd: vx, vy, walking height, vyaw
}

# config for gaitlibary_test.py
config_static = {
    "max_timesteps": 2500,  # NOTE: max timestep per episode
    "is_visual": True,  # NOTE: set true to visualize robot
    "cam_track_robot": False,  # NOTE: set true to let camera track the robot
    "step_zerotorque": True,  # NOTE: set true to step without torque
    "minimal_rand": True,  # NOTE: set true to use minimal rand range -> only rand floor friction
    "is_noisy": False,  # NOTE: set true to add noise on observation
    "add_perturbation": False,  # NOTE: set true to add external perturbation
    "add_standing": False,  # NOTE: set true to include standing skill
    "fixed_gait": True,  # NOTE: set true to only test one cmd of 'fixed_gait_cmd'
    "add_rotation": False,  # NOTE: set true to add rotation (vyaw) command, only useful if not fixed gait
    "fixed_gait_cmd": [
        0.0,
        0.0,
        0.98,
        0.0,
    ],  # fixed gait cmd: vx, vy, walking height, vyaw
}

# config for jumping training stages
config_jump_stage1 = {
    "max_timesteps": 2500,
    "is_visual": False,
    "stage": 1,  # Initial stabilization stage
    "minimal_rand": True,  # Limited randomization in stage 1
    "is_noisy": True,
    "add_perturbation": False,
    "add_standing": True,
    "fixed_gait": True,
    "add_rotation": False,
    "step_zerotorque": False,
    "cam_track_robot": False,
    "thruster_penalty_scale": 0.1,  # High penalty to discourage thruster use in stage 1
    "fixed_gait_cmd": [0.0, 0.0, 0.98, 0.0],  # Stand in place
}

config_jump_stage2 = {
    "max_timesteps": 2500,
    "is_visual": False,
    "stage": 2,  # Jump execution stage
    "minimal_rand": True,
    "is_noisy": True,
    "add_perturbation": False,
    "add_standing": False,  # Focus on jumping
    "fixed_gait": False,  # Allow variable height jumps
    "add_rotation": False,
    "step_zerotorque": False,
    "cam_track_robot": False,
    "thruster_penalty_scale": 0.01,  # Lower penalty to allow thruster use
    "target_jump_height": 0.3,  # Initial target jump height (meters)
}

config_jump_stage3 = {
    "max_timesteps": 2500,
    "is_visual": False,
    "stage": 3,  # Recovery and robustness stage
    "minimal_rand": False,  # Full dynamics randomization
    "is_noisy": True,
    "add_perturbation": True,  # Add external perturbations
    "add_standing": True,  # Mix standing and jumping
    "fixed_gait": False,
    "add_rotation": True,  # Allow directional jumps
    "step_zerotorque": False,
    "cam_track_robot": False,
    "thruster_penalty_scale": 0.01,
    "target_jump_height": 0.5,  # Increased jump height
    "randomize_targets": True,  # Randomize jump targets
}
