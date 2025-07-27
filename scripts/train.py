import os
from baselines.common import tf_util as U
import tensorflow as tf
import numpy as np

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
from ppo.policies import MLPCNNPolicy
from rlenv.cassie_env import CassieEnv
from configs.defaults import ROOT_PATH
from baselines import logger
from mpi4py import MPI
import argparse

os.environ["OPENAI_LOG_FORMAT"] = "stdout,log,tensorboard"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

model_folder = ROOT_PATH + "/ckpts/"

def get_args():
    parser = argparse.ArgumentParser(description="training_setup")
    parser.add_argument(
        "--train_name", type=str, default="test", help="naming the training"
    )
    parser.add_argument(
        "--rnd_seed", type=int, default=42, help="random seed (default 42)"
    )
    parser.add_argument(
        "--max_iters", type=int, default=5000, help="max iterations per stage (default 5000)"
    )
    parser.add_argument(
        "--restore_from",
        type=str,
        default=None,
        help="restore_from previous checkpoint (default None)",
    )
    parser.add_argument(
        "--restore_cont",
        type=int,
        default=None,
        help="counts that has been resumed (default None)",
    )
    parser.add_argument(
        "--save_interval",
        type=int,
        default=100,
        help="save the ckpt every save_interval iteration",
    )
    # New arguments for jumping curriculum
    parser.add_argument(
        "--curriculum", 
        action="store_true",
        help="Enable curriculum learning for jumping"
    )
    parser.add_argument(
        "--stage",
        type=int,
        choices=[1, 2, 3],
        help="Specific stage to train (1: stabilization, 2: jumping, 3: recovery)"
    )
    parser.add_argument(
        "--gpu",
        type=int,
        default=None,
        help="GPU device ID to use (None for CPU)"
    )
    
    args = parser.parse_args()
    return args

def train(env, policy_fn, max_iters=5000, continue_from=None, callback=None, **kwargs):
    """Generic training function that supports curriculum learning"""
    
    # Configure GPU/CPU
    if kwargs.get('gpu') is None:
        config = tf.ConfigProto(device_count={"GPU": 0})
        U.make_session(config=config).__enter__()
        print("**************Using CPU**************")
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(kwargs.get('gpu'))
        U.make_session().__enter__()
        print(f"**************Using GPU {kwargs.get('gpu')}**************")

    from ppo import ppo_sgd_cnn as ppo_sgd

    pi = ppo_sgd.learn(
        env,
        policy_fn,
        max_iters=max_iters,
        timesteps_per_actorbatch=4096,
        clip_param=0.2,
        entcoeff=0,
        optim_epochs=2,
        optim_stepsize=1e-4,
        optim_batchsize=512,
        gamma=0.98,
        lam=0.95,
        callback=callback,
        schedule="constant",
        continue_from=continue_from,
    )
    return pi

def train_jumping_curriculum(args):
    """Implements the three-stage curriculum for thruster-assisted jumping"""
    from configs.env_config import config_jump_stage1, config_jump_stage2, config_jump_stage3
    
    def policy_fn(name, ob_space_vf, ob_space_pol, ob_space_pol_cnn, ac_space):
        return MLPCNNPolicy(
            name=name,
            ob_space_vf=ob_space_vf,
            ob_space_pol=ob_space_pol,
            ob_space_pol_cnn=ob_space_pol_cnn,
            ac_space=ac_space,
            hid_size=512,
            num_hid_layers=2,
        )

    stages = [
        ("stabilization", config_jump_stage1),
        ("jumping", config_jump_stage2),
        ("recovery", config_jump_stage3)
    ]

    policy = None
    for stage_idx, (stage_name, config) in enumerate(stages, 1):
        if args.stage and stage_idx != args.stage:
            continue
            
        print(f"\n=== Starting Stage {stage_idx}: {stage_name} ===")
        
        # Create environment with stage-specific config
        env = CassieEnv(config=config)
        
        # Setup checkpoint paths
        stage_model_name = f"{args.train_name}_stage{stage_idx}"
        if args.restore_cont:
            stage_model_name += f"_cont{args.restore_cont}"
        
        checkpoint_dir = os.path.join(model_folder, stage_model_name)
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        # Configure logging
        log_dir = os.path.join(ROOT_PATH, "logs", stage_model_name)
        os.environ["OPENAI_LOGDIR"] = log_dir
        logger.configure()

        def stage_callback(locals_, globals_):
            """Callback for saving checkpoints and logging stage-specific metrics"""
            if MPI.COMM_WORLD.Get_rank() != 0:
                return True
                
            iters_so_far = locals_["iters_so_far"]
            if iters_so_far % args.save_interval == 0:
                saver = locals_["saver"]
                sess = U.get_session()
                checkpoint_path = os.path.join(checkpoint_dir, "model")
                saver.save(sess, checkpoint_path, global_step=locals_["timesteps_so_far"])
                
            # Log stage-specific metrics
            if "info" in locals_ and "reward_dict" in locals_["info"]:
                reward_dict = locals_["info"]["reward_dict"]
                
                # Log all reward components
                for reward_name, value in reward_dict.items():
                    logger.logkv(f"stage{stage_idx}/{reward_name}", value)
                
                # Log thruster-specific metrics
                if "thruster_sum" in reward_dict:
                    logger.logkv(f"thruster/total_force", reward_dict["thruster_sum"])
                    logger.logkv(f"thruster/mean_force", reward_dict["thruster_mean"])
                    logger.logkv(f"thruster/max_force", reward_dict["thruster_max"])
                
                # Log jump height during airborne stage
                if stage_idx == 2 and "jump_height" in reward_dict:
                    logger.logkv("jump/height", reward_dict["jump_height"])
                
            logger.dumpkvs()
            return True

        # Train the stage
        policy = train(
            env=env,
            policy_fn=policy_fn,
            max_iters=args.max_iters,
            continue_from=policy,  # Pass previous policy for curriculum
            callback=stage_callback,
            gpu=args.gpu
        )
        
        print(f"=== Completed Stage {stage_idx}: {stage_name} ===")
        
        # Save final stage checkpoint
        if MPI.COMM_WORLD.Get_rank() == 0:
            saver = tf.train.Saver()
            sess = U.get_session()
            final_checkpoint_path = os.path.join(checkpoint_dir, "model_final")
            saver.save(sess, final_checkpoint_path)

    return policy

if __name__ == "__main__":
    args = get_args()
    
    # Set random seed if specified
    if args.rnd_seed is not None:
        np.random.seed(args.rnd_seed)
        tf.set_random_seed(args.rnd_seed)
    
    if args.curriculum or args.stage:
        # Run curriculum learning
        train_jumping_curriculum(args)
    else:
        # Run regular training with config_train
        from configs.env_config import config_train
        
        def policy_fn(name, ob_space_vf, ob_space_pol, ob_space_pol_cnn, ac_space):
            return MLPCNNPolicy(
                name=name,
                ob_space_vf=ob_space_vf,
                ob_space_pol=ob_space_pol,
                ob_space_pol_cnn=ob_space_pol_cnn,
                ac_space=ac_space,
                hid_size=512,
                num_hid_layers=2,
            )
        
        env = CassieEnv(config=config_train)
        
        # Configure logging
        if args.restore_cont and args.restore_from:
            saved_model = f"{args.train_name}_rnds{args.rnd_seed}_cont{args.restore_cont}"
        else:
            saved_model = f"{args.train_name}_rnds{args.rnd_seed}"
            
        os.environ["OPENAI_LOGDIR"] = os.path.join(ROOT_PATH, "logs", saved_model)
        logger.configure()
        
        def training_callback(locals_, globals_):
            if MPI.COMM_WORLD.Get_rank() == 0 and locals_["iters_so_far"] % args.save_interval == 0:
                saver = locals_["saver"]
                sess = U.get_session()
                model_dir = os.path.join(model_folder, saved_model)
                os.makedirs(model_dir, exist_ok=True)
                saver.save(sess, os.path.join(model_dir, "model"), global_step=locals_["timesteps_so_far"])
            return True
        
        train(
            env=env,
            policy_fn=policy_fn,
            max_iters=args.max_iters,
            continue_from=args.restore_from,
            callback=training_callback,
            gpu=args.gpu
        )