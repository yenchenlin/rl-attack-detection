import tensorflow as tf
import numpy as np
import cv2

import argparse
import sys, os
import logging

import cPickle as pickle

from model import ActionConditionalVideoPredictionModel
from dataset import Dataset, CaffeDataset
from util import post_process

def get_config(args):
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    return config

def load_caffe_model(x, path):
    with open(path, 'rb') as f:
        data = pickle.load(f)
        w = tf.Variable(data['conv1/w'], dtype=tf.float32)
        b = tf.Variable(data['conv1/b'], dtype=tf.float32)
        l = tf.nn.conv2d(x, w, strides=[1, 2, 2, 1], padding='VALID', name='conv2d')
        l = tf.nn.bias_add(l, b, name='bias_add')
    return l

def main(args):
    with tf.Graph().as_default() as graph:
        # Create dataset
        logging.info('Create data flow from %s' % args.data)
        caffe_dataset = CaffeDataset(dir=args.data, num_act=args.num_act, mean_path=args.mean)
       
        # Config session
        config = get_config(args)

        x = tf.placeholder(dtype=tf.float32, shape=[None, 84, 84, 12])
        op = load_caffe_model(x, args.load)

        init = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
        
        # Start session
        with tf.Session(config=config) as sess:
            sess.run(init)
            i = 0
            for s, a in caffe_dataset(5):
                pred_data = sess.run([op], feed_dict={x: [s]})[0]
                print pred_data.shape
                np.save('tf-%03d.npy' % i, pred_data)
                i += 1
           
if __name__ == '__main__':
    logging.basicConfig(format='[%(asctime)s] %(message)s', datefmt='%m/%d/%Y %I:%M:%S %p', level=logging.INFO)
    parser = argparse.ArgumentParser()
    parser.add_argument('--log', help='summary directory', type=str, default='caffe-test')
    parser.add_argument('--data', help='testing data directory', type=str, required=True)
    parser.add_argument('--mean', help='image mean path', type=str, required=True)
    parser.add_argument('--load', help='caffe-dumped model path', type=str, required=True)
    parser.add_argument('--num_act', help='num acts', type=int, required=True)
    args = parser.parse_args()

    main(args)



