import argparse
import gym
import os
import numpy as np
import tensorflow as tf
import cv2
import glob
from collections import deque

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

from scipy.stats import entropy
from baselines.deepq.experiments.atari.squeeze import median_filter_np, binary_filter_np


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


def test_gen(sess, obs, act, gt, mean, model, n_actions, step):
    obs = obs - mean[None]
    obs = obs * 1/255.0
    onehot_act = np.zeros((1, n_actions))
    onehot_act[0, act] = 1

    pred = model.predict(sess, obs, onehot_act)[0]

    pred = pred * 255.0
    pred = pred + mean[None]
    #print(gt[:, :, -1].shape, pred.shape)
    #print(np.sum(gt[:, :, -1][:, :, np.newaxis] - pred[0, :, :, :]))
    #cv2.imwrite('/home/yclin/viz/gt_{}.png'.format(step), gt[:, :, -1][:, :, np.newaxis])
    #cv2.imwrite('/home/yclin/viz/pred_{}.png'.format(step), pred[0, :, :, :])
    return pred[0, :, :, 0]


def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    return np.exp(x) / np.sum(np.exp(x), axis=0)


def get_newest_model(path):
    models = glob.glob(os.path.join(path, 'model.ckpt-*.meta'))
    newest_model = max(models, key=os.path.getctime)
    return newest_model[:-5]


def play(env, acts, stochastic, video_path, game_name, attack=None, q_func=None):
    act = acts[0]
    if attack != None:
        adv_act = acts[1]
        adv_env = acts[2]

    num_episodes = 0
    step = 0
    video_recorder = None
    video_recorder = VideoRecorder(
        env, video_path, enabled=video_path is not None)
    obs = env.reset()
    pred_obs = deque(maxlen=4)
    sess = U.get_session()

    thresholds = [0.1, 0.2, 0.5, 0.6, 0.7, 0.8, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 10.0, 13.0, 15.0, 20.0, 25.0, 30.0, 35.0, 40.0, 45.0, 50.0, 55.0, 60.0, 70, 80, 90]
    #thresholds = [0.0]
    detection = 0.0
    attack_count = 0.0
    attack_success = 0.0
    #fp = 0.0
    #tp = 0.0
    fp = {key: 0 for key in thresholds}
    tp = {key: 0 for key in thresholds}

    filter_size = 2
    fp_sq = {key: 0 for key in thresholds}
    tp_sq = {key: 0 for key in thresholds}


    if attack != None:
        from baselines.deepq.prediction.tfacvp.model import ActionConditionalVideoPredictionModel
        gen_dir = '/home/yclin/Workspace/rl-adversarial-attack-detection/baselines/baselines/deepq/prediction'
        # Automatically load the newest model
        model_path = get_newest_model(os.path.join(gen_dir, 'models/{}/train'.format(game_name)))
        mean_path = os.path.join(gen_dir, '{}_episodes/mean.npy'.format(game_name))

        mean = np.load(mean_path)
        with tf.variable_scope('G'):
            model = ActionConditionalVideoPredictionModel(num_act=env.action_space.n, num_channel=1, is_train=False)
            model.restore(sess, model_path, 'G')

    if attack == 'carliniL2':
        carliniLi = CarliniLi(sess, q_func, env.action_space.n)

    while True:
        #env.unwrapped.render()
        #video_recorder.capture_frame()
        step += 1
        if attack == 'carliniL2':
            obs = (np.array(obs) - 127.5) / 255.0
            adv_obs = carliniLi.attack_single(obs, [0., 0., 0., 0., 1., 0.])
            adv_obs = np.array(adv_obs) * 255.0 + 127.5
            action = act(np.array(adv_obs), stochastic=stochastic)[0]
            print(action)
            obs, rew, done, info = env.step(action)
        elif attack == 'fgsm':
            # np.array(obs)[None]: (1, 84, 84, 4)
            adv_q_values = adv_act(np.array(obs)[None], stochastic=stochastic)[0]
            adv_action = np.argmax(adv_q_values)
            q_values = act(np.array(obs)[None], stochastic=stochastic)[0]
            action = np.argmax(q_values)

            # Adversary determine whether to attack
            is_attack = True if np.random.rand() < 0.5 else False
            if is_attack:
                attack_count += 1

            # Decide final action and q_values
            final_act = adv_action if is_attack else action
            final_q_values = adv_q_values if is_attack else q_values

            # Get squeezed input
            adv_obs = adv_env(np.array(obs)[None], stochastic=stochastic)[0] * 255.0
            final_obs = adv_obs if is_attack else np.array(obs)
            sq_final_obs = median_filter_np(final_obs, filter_size)
            sq_final_q_values = act(np.array(sq_final_obs)[None], stochastic=stochastic)[0]

            # Defensive planning
            if step >= 4:
                final_old_obs = adv_env(np.array(old_obs)[None], stochastic=stochastic)[0] * 255.0 if is_attack else old_obs
                pred_obs.append(test_gen(sess, final_old_obs, old_action, np.array(obs), mean, model,
                    env.action_space.n, step))
                if len(pred_obs) == 4:
                    # Feed pred state into policy
                    pred_q_values = act(np.stack(pred_obs, axis=2)[None], stochastic=stochastic)[0]
                    pred_act = np.argmax(pred_q_values)

                    # Attack success
                    if adv_action != action and is_attack:
                        attack_success += 1

                    # Detect as attack
                    for threshold in thresholds:
                        # Feature squeezer
                        if np.sum(np.abs(sq_final_q_values - final_q_values)) > threshold:
                            #final_act = np.argmax(sq_final_q_values)
                            if is_attack:
                                tp_sq[threshold] += 1
                            else:
                                fp_sq[threshold] += 1

                        # Ours
                        if np.sum(np.abs(pred_q_values - final_q_values)) > threshold:
                        #if np.sum(np.abs(pred_q_values - final_q_values)) > 0.1:
                            final_act = action
                            # False positive
                            #if not is_attack or adv_action == action:
                            if not is_attack:
                                #print(entropy(pk=softmax(pred_q_values), qk=softmax(final_q_values)))
                                fp[threshold] += 1
                            # True positive
                            #if is_attack and adv_action != action:
                            if is_attack and adv_action != action:
                                tp[threshold] += 1

            old_obs = np.array(obs)
            old_action = adv_action
            obs, rew, done, info = env.step(final_act)
        else:
            action = act(np.array(obs)[None], stochastic=stochastic)[0]
            obs, rew, done, info = env.step(action)

        if done: # Not end of an episode
            obs = env.reset()

        # End of an episode
        if len(info["rewards"]) > num_episodes:
            if len(info["rewards"]) == 1 and video_recorder.enabled:
                # save video of first episode
                print("Saved video.")
                video_recorder.close()
                video_recorder.enabled = False
            print(info["rewards"][-1])
            num_episodes = len(info["rewards"])

            if attack != None:
                print("Attack ratio:", attack_success/step)
                if attack_count != 0:
                    print("Attack success rate:", attack_success/attack_count)
                    #print("Precision:", tp/(tp+fp))
                    #print("False Positive:", fp/(tp+fp))
                    #print("Recall:", tp/attack_success)
                    #print("Step: ", step, "Number of TP: ", tp, "Number of FP:", fp)

                for t in thresholds:
                    print(t)
                    print("Precision:", tp[t]/(tp[t]+fp[t]))
                    print("False Positive:", fp[t]/(tp[t]+fp[t]))
                    print("Recall:", tp[t]/attack_success)
                    print("Step: ", step, "Number of TP: ", tp[t], "Number of FP:", fp[t])

                """
                print("Feature Squeezer")
                for t in thresholds:
                    print(t)
                    print("Precision:", tp_sq[t]/(tp_sq[t]+fp_sq[t]))
                    print("False Positive:", fp_sq[t]/(tp_sq[t]+fp_sq[t]))
                    print("Recall:", tp_sq[t]/attack_success)
                    print("Step: ", step, "Number of TP: ", tp_sq[t], "Number of FP:", fp_sq[t])
                """
            step = 0
            exit()

if __name__ == '__main__':
    with U.make_session(4) as sess:
        args = parse_args()
        env = make_env(args.env)
        q_func = dueling_model if args.dueling else model
        acts = deepq.build_act(
            make_obs_ph=lambda name: U.Uint8Input(env.observation_space.shape, name=name),
            q_func=q_func,
            num_actions=env.action_space.n,
            attack=args.attack)
        U.load_state(os.path.join(args.model_dir, "saved"))
        play(env, acts, args.stochastic, args.video, args.env, args.attack, q_func=q_func)
