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
from baselines.deepq.prediction.tool.episode_collector import EpisodeCollector

import pdb


def parse_args():
    parser = argparse.ArgumentParser("Run an already learned DQN model.")
    # Environment
    parser.add_argument("--env", type=str, required=True, help="name of the game")
    parser.add_argument("--model-dir", type=str, default=None, help="load model from this directory. ")
    parser.add_argument("--video", type=str, default=None, help="Path to mp4 file where the video of first episode will be recorded.")
    boolean_flag(parser, "stochastic", default=True, help="whether or not to use stochastic actions according to models eps value")
    boolean_flag(parser, "dueling", default=False, help="whether or not to use dueling model")

    return parser.parse_args()


def make_env(game_name):
    env = gym.make(game_name + "NoFrameskip-v4")
    env = SimpleMonitor(env)
    env = wrap_dqn(env)
    return env


def play(env, act, stochastic, video_path):
    # Collector
    directory = 'PongNoFrameskip-v4'
    preprocess_func = lambda x: x
    n_episodes = 200
    episode = 0
    if not os.path.exists(directory):
        os.makedirs(directory)
    collector = EpisodeCollector(path=os.path.join(directory, '%04d.tfrecords' % (episode)), preprocess_func=preprocess_func, skip=0)

    timestep = 0
    num_episodes = 0
    video_recorder = None
    video_recorder = VideoRecorder(
        env, video_path, enabled=video_path is not None)
    obs = env.reset()
    while True:
        #env.unwrapped.render()
        #video_recorder.capture_frame()
        action = act(np.array(obs)[None], stochastic=stochastic)[0]
        prev_obs = obs
        obs, rew, done, info = env.step(action)
        #if timestep == 100:
        #    pdb.set_trace()
        #    print(np.array_equal(np.array(prev_obs)[-1], np.array(obs)[-1]))
        timestep += 1

        collector.save(s=np.array(prev_obs), a=action, x_next=np.array(obs)[:, :, -1])
        if done:
            obs = env.reset()
            timestep = 0

            collector.close()
            episode += 1
            print("Episode: {}".format(episode))
            if episode == n_episodes:
                break
            collector = EpisodeCollector(path=os.path.join('PongNoFrameskip-v4', '%04d.tfrecords' % (episode)),
                        preprocess_func=preprocess_func, skip=0)

        if len(info["rewards"]) > num_episodes:
            if len(info["rewards"]) == 1 and video_recorder.enabled:
            # save video of first episode
                print("Saved video.")
                video_recorder.close()
                video_recorder.enabled = False
            print(info["rewards"][-1])
            num_episodes = len(info["rewards"])


if __name__ == '__main__':
    with U.make_session(4) as sess:
        args = parse_args()
        env = make_env(args.env)
        act = deepq.build_act(
            make_obs_ph=lambda name: U.Uint8Input(env.observation_space.shape, name=name),
            q_func=dueling_model if args.dueling else model,
            num_actions=env.action_space.n)
        U.load_state(os.path.join(args.model_dir, "saved"))
        play(env, act, args.stochastic, args.video)
