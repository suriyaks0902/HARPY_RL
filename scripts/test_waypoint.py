import os
import tensorflow as tf
from baselines.common import tf_util as U
from ppo.policies import MLPCNNPolicy
from rlenv.cassie_env import CassieEnv
from configs.defaults import ROOT_PATH
from configs.env_config import config_play

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # Use CPU

def main():
    """
    Test waypoint navigation and jumping with a trained model
    """
    env = CassieEnv(config=config_play)
    
    # Set up policy
    ob_space_pol = env.observation_space_pol
    ac_space = env.action_space
    ob_space_vf = env.observation_space_vf
    ob_space_pol_cnn = env.observation_space_pol_cnn
    
    pi = MLPCNNPolicy(
        name="pi",
        ob_space_vf=ob_space_vf,
        ob_space_pol=ob_space_pol,
        ob_space_pol_cnn=ob_space_pol_cnn,
        ac_space=ac_space,
        hid_size=512,
        num_hid_layers=2,
    )

    # Load trained model
    config = tf.ConfigProto(device_count={"GPU": 0})
    U.make_session(config=config).__enter__()
    
    model_path = os.path.join(ROOT_PATH, "ckpts/versatile_walking")
    model_path = tf.train.latest_checkpoint(model_path)
    if model_path is None:
        raise ValueError("❌ Could not find trained model")
    
    U.load_state(model_path)
    print(f"✓ Loaded model from {model_path}")

    # Test loop
    draw_state = env.render()
    episode = 0
    
    while draw_state and episode < 5:  # Run 5 episodes
        obs_vf, obs_pol = env.reset()
        total_reward = 0
        steps = 0
        
        print(f"\nEpisode {episode + 1}")
        print(f"Target waypoint: {env.waypoints[env.current_waypoint_idx]}")
        
        while steps < env.max_timesteps:
            if not env.vis.ispaused():
                # Get action from policy
                ac = pi.act(stochastic=False, ob_vf=obs_vf, ob_pol=obs_pol)[0]
                
                # Step environment
                obs_vf, obs_pol, reward, done, info = env.step(ac)
                total_reward += reward
                
                # Print status
                curr_pos = env.qpos[:3]
                target_pos = env.waypoints[env.current_waypoint_idx]
                dist = ((curr_pos[0] - target_pos[0])**2 + 
                       (curr_pos[1] - target_pos[1])**2 + 
                       (curr_pos[2] - target_pos[2])**2)**0.5
                
                print(f"\rStep {steps}: Pos {curr_pos.round(2)}, "
                      f"Target {target_pos}, Distance {dist:.2f}, "
                      f"Reward {reward:.2f}", end="")
                
                # Render
                draw_state = env.render()
                steps += 1
                
                if done:
                    break
            else:
                while env.vis.ispaused() and draw_state:
                    draw_state = env.render()
        
        print(f"\nEpisode {episode + 1} finished")
        print(f"Total steps: {steps}")
        print(f"Total reward: {total_reward:.2f}")
        episode += 1

if __name__ == "__main__":
    main() 