import argparse
import gym
import os
import numpy as np

from gym.monitoring import VideoRecorder

import baselines.common.tf_util as U

from baselines import deepq
from baselines.common.misc_util import (
    boolean_flag,
    SimpleMonitor,
)
from baselines.common.atari_wrappers_deprecated import wrap_dqn
from baselines.deepq.experiments.atari.model import model, dueling_model
from baselines.deepq.li_attack import CarliniLi


def parse_args():
    parser = argparse.ArgumentParser("Run an already learned DQN model.")
    # Environment
    parser.add_argument("--env", type=str, required=True, help="name of the game")
    parser.add_argument("--model-dir", type=str, default=None, help="load model from this directory. ")
    parser.add_argument("--video", type=str, default=None, help="Path to mp4 file where the video of first episode will be recorded.")
    boolean_flag(parser, "stochastic", default=True, help="whether or not to use stochastic actions according to models eps value")
    boolean_flag(parser, "dueling", default=False, help="whether or not to use dueling model")
    parser.add_argument("--attack", type=str, default=None, help="Method to attack the model.")

    return parser.parse_args()


def make_env(game_name):
    env = gym.make(game_name + "NoFrameskip-v4")
    env = SimpleMonitor(env)
    env = wrap_dqn(env)
    return env


def play(env, act, stochastic, video_path, attack=None, q_func=None):
    num_episodes = 0
    step = 0
    video_recorder = None
    video_recorder = VideoRecorder(
        env, video_path, enabled=video_path is not None)
    obs = env.reset()

    if attack == 'carliniL2':
        sess = U.get_session()
        carliniLi = CarliniLi(sess, q_func, env.action_space.n)

    while True:
        #env.unwrapped.render()
        video_recorder.capture_frame()

        if attack == 'carliniL2':
            step += 1
            obs = (np.array(obs) - 127.5) / 255.0
            adv_obs = carliniLi.attack_single(obs, [0., 0., 0., 0., 1., 0.])
            print(np.min(adv_obs))
            adv_obs = np.array(adv_obs) * 255.0 + 127.5
            action = act(np.array(adv_obs), stochastic=stochastic)[0]
            obs, rew, done, info = env.step(action)
        else:
            action = act(np.array(obs)[None], stochastic=stochastic)[0]
            obs, rew, done, info = env.step(action)

        print("Step: {}".format(step))
        print("Action: {}".format(action))

        if done:
            obs = env.reset()
        if len(info["rewards"]) > num_episodes:
            if len(info["rewards"]) == 1 and video_recorder.enabled:
                # save video of first episode
                print("Saved video.")
                video_recorder.close()
                video_recorder.enabled = False
            print(info["rewards"][-1])
            num_episodes = len(info["rewards"])
            step = 0


if __name__ == '__main__':
    with U.make_session(4) as sess:
        args = parse_args()
        env = make_env(args.env)
        q_func = dueling_model if args.dueling else model
        act = deepq.build_act(
            make_obs_ph=lambda name: U.Uint8Input(env.observation_space.shape, name=name),
            q_func=q_func,
            num_actions=env.action_space.n, 
            attack=args.attack)
        U.load_state(os.path.join(args.model_dir, "saved"))
        play(env, act, args.stochastic, args.video, args.attack, q_func=q_func)
