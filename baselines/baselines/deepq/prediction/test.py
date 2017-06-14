import tensorflow as tf
import numpy as np
import cv2

import argparse
import sys, os
import logging

import cPickle as pickle

from tfacvp.model import ActionConditionalVideoPredictionModel
from tfacvp.dataset import Dataset, CaffeDataset
from tfacvp.util import post_process

def get_config(args):
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    return config

def load_caffe_model(path):
    tf_ops = []
    with tf.variable_scope('', reuse=True) as scope:
        with open(path, 'rb') as f:
            data = pickle.load(f)
            for key in data:
                val = data[key]
                var = tf.get_variable(key)
                tf_ops.append(tf.assign(var, data[key]))
                logging.info('%s loaded with shape %s' % (key, val.shape))
    return tf.group(*tf_ops)

def main(args):
    with tf.Graph().as_default() as graph:
        # Create dataset
        logging.info('Create data flow from %s' % args.data)
        caffe_dataset = CaffeDataset(dir=args.data, num_act=args.num_act, mean_path=args.mean)
        
        # Create model
        logging.info('Create model from %s' % (args.load))
        model = ActionConditionalVideoPredictionModel(inputs=None, num_act=args.num_act, is_train=False)

        # Create initializer
        init = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
        
        # Create weight load operation
        load_op = load_caffe_model(args.load)
         
        # Config session
        config = get_config(args)
        
        # Start session
        with tf.Session(config=config) as sess:
            logging.info('Loading')
            sess.run(load_op)
            op = graph.get_tensor_by_name(args.layer)
            i = 0
            for s, a in caffe_dataset(5):
                pred_data = sess.run([op], feed_dict={model.inputs['s_t']: [s],
                                                                model.inputs['a_t']: a})[0]
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
    parser.add_argument('--layer', help='output layer', type=str, required=True)
    args = parser.parse_args()

    main(args)



