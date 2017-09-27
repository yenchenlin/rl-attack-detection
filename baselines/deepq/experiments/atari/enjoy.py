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
import tensorflow as tf
import cv2
from collections import deque


def parse_args():
    parser = argparse.ArgumentParser("Run an already learned DQN model.")
    # Environment
    parser.add_argument("--env", type=str, required=True, help="name of the game")
    parser.add_argument("--model-dir", type=str, default=None, help="load model from this directory. ")
    parser.add_argument("--video", type=str, default=None, help="Path to mp4 file where the video of first episode will be recorded.")
    boolean_flag(parser, "stochastic", default=True, help="whether or not to use stochastic actions according to models eps value")
    boolean_flag(parser, "dueling", default=False, help="whether or not to use dueling model")
    parser.add_argument("--attack", type=str, default=None, help="Method to attack the model.")
    parser.add_argument("--defense", type=str, default=None, help="Method to defend the attack.")

    return parser.parse_args()


def make_env(game_name):
    env = gym.make(game_name + "NoFrameskip-v4")
    env = SimpleMonitor(env)
    env = wrap_dqn(env)
    return env


def load_visual_foresight(game_name):
    sess = U.get_session()
    from baselines.deepq.prediction.tfacvp.model import ActionConditionalVideoPredictionModel
    gen_dir = './atari-visual-foresight/'
    model_path = os.path.join(gen_dir, '{}/model.ckpt'.format(game_name))
    mean_path = os.path.join(gen_dir, '{}/mean.npy'.format(game_name))
    game_screen_mean = np.load(mean_path)
    with tf.variable_scope('G'):
        foresight = ActionConditionalVideoPredictionModel(num_act=env.action_space.n, num_channel=1, is_train=False)
        foresight.restore(sess, model_path, 'G')
    return foresight, game_screen_mean


def foresee(sess, obs, act, gt, mean, model, n_actions, step):
    onehot_act = np.zeros((1, n_actions))
    onehot_act[0, act] = 1
    obs = obs - mean[None]
    obs = obs * 1/255.0
    pred_frame = model.predict(sess, obs, onehot_act)[0]
    pred_frame = pred_frame* 255.0
    pred_frame = pred_frame + mean[None]
    #print(gt[:, :, -1].shape, pred_frame.shape)
    #print(np.sum(gt[:, :, -1][:, :, np.newaxis] - pred_frame[0, :, :, :]))
    #cv2.imwrite('./tmp/gt_{}.png'.format(step), gt[:, :, -1][:, :, np.newaxis])
    #cv2.imwrite('./tmp/pred_{}.png'.format(step), pred_frame[0, :, :, :])
    return pred_frame[0, :, :, 0]


def play(env, act, craft_adv_obs, stochastic, video_path, game_name, attack, defense):
    if defense == 'foresight':
        vf, game_screen_mean = load_visual_foresight(game_name)
        pred_obs = deque(maxlen=4)

    num_episodes = 0
    video_recorder = None
    video_recorder = VideoRecorder(
        env, video_path, enabled=video_path is not None)

    t = 0
    obs = env.reset()
    while True:
        #env.unwrapped.render()
        video_recorder.capture_frame()

	# Attack
        if craft_adv_obs != None:
            # Craft adv. examples
            adv_obs = craft_adv_obs(np.array(obs)[None], stochastic=stochastic)[0]
            action = act(np.array(adv_obs)[None], stochastic=stochastic)[0]
        else:
            # Normal
            action = act(np.array(obs)[None], stochastic=stochastic)[0]

	# Defense
        if t > 4 and defense == 'foresight':
            pred_obs.append(
                foresee(U.get_session(), old_obs, old_action, np.array(obs), game_screen_mean, vf,
                        env.action_space.n, t)
            )
            if len(pred_obs) == 4:
                action = act(np.stack(pred_obs, axis=2)[None], stochastic=stochastic)[0]

        old_obs = obs
        old_action = action

        # RL loop
        obs, rew, done, info = env.step(action)
        t += 1
        if done:
            t = 0
            obs = env.reset()
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
        # Build graph and load agents
        act, craft_adv_obs = deepq.build_act(
            make_obs_ph=lambda name: U.Uint8Input(env.observation_space.shape, name=name),
            q_func=dueling_model if args.dueling else model,
            num_actions=env.action_space.n,
            attack=args.attack,
            model_path=os.path.join(args.model_dir, "saved")
        )
        play(env, act, craft_adv_obs, args.stochastic, args.video, args.env, args.attack, args.defense)
