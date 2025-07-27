import numpy as np
from rlenv.cassie_env import CassieEnv
from configs.env_config import config_jump_stage3
from baselines.common import tf_util as U
import tensorflow as tf

def test_jumping(policy_path, num_episodes=10, render=True):
    """
    Test the trained jumping policy
    """
    # Load environment with visualization
    config = config_jump_stage3.copy()
    config["is_visual"] = render
    config["cam_track_robot"] = True
    env = CassieEnv(config)
    
    # Load policy
    sess = tf.Session()
    with sess.as_default():
        saver = tf.train.import_meta_graph(policy_path + ".meta")
        saver.restore(sess, policy_path)
    
    for episode in range(num_episodes):
        obs = env.reset()
        done = False
        episode_reward = 0
        max_height = 0
        
        while not done:
            # Get action from policy
            action = policy.predict(obs)
            
            # Step environment
            obs, reward, done, info = env.step(action)
            
            # Track metrics
            episode_reward += reward
            max_height = max(max_height, env.qpos[2])
            
            if render:
                env.render()
        
        print(f"Episode {episode+1}")
        print(f"Total Reward: {episode_reward:.2f}")
        print(f"Max Height: {max_height:.2f}m")
        print("------------------------")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy", type=str, required=True, help="Path to policy checkpoint")
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--no-render", action="store_true")
    args = parser.parse_args()
    
    test_jumping(
        policy_path=args.policy,
        num_episodes=args.episodes,
        render=not args.no_render
    ) 