from baselines.common import tf_util as U
import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""
import tensorflow as tf
from rlenv.cassie_env import CassieEnv
import ppo.policies as policies
from configs.env_config import config_play
from configs.defaults import ROOT_PATH
import numpy as np

# to ignore specific deprecation warnings
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
import argparse

model_folder = ROOT_PATH + "/ckpts/"

def get_args():
    parser = argparse.ArgumentParser(description="testing_setup")

    parser.add_argument(
        "--test_model", type=str, default=None, help="checkpoint to test (default None)"
    )
    parser.add_argument(
        "--test_episode_len", type=int, default=20000, help="episode length to test -- can be longer than the env episode length"
    )
    args = parser.parse_args()

    return args


args = get_args()
# model name to test
test_model = args.test_model
test_episode_len = args.test_episode_len


def main():
    config = tf.ConfigProto(device_count={"GPU": 0})
    U.make_session(config=config)

    env = CassieEnv(config=config_play)
    draw_state = env.render()
    obs_vf, obs_pol = env.reset()

    while draw_state:
        if not env.vis.ispaused():
            dummy_action = np.zeros(env.action_space.shape)
            obs_vf, obs_pol, reward, done, info = env.step(dummy_action)
            draw_state = env.render()
            if done:
                obs_vf, obs_pol = env.reset()
        else:
            while env.vis.ispaused() and draw_state:
                draw_state = env.render()




if __name__ == "__main__":
    main()
