import tensorflow as tf
import numpy as np
import cv2

import argparse
import sys, os
import logging

from model import ActionConditionalVideoPredictionModel
from dataset import Dataset

def get_config(args):
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    return config

def main(args):
    with tf.Graph().as_default() as graph:   
        model = ActionConditionalVideoPredictionModel(num_act=args.num_act)
        for var in tf.trainable_variables():
            print var
        with tf.variable_scope('', reuse=True) as scope:
            print tf.get_variable('conv1/w')
        
    
if __name__ == '__main__':
    logging.basicConfig(format='[%(asctime)s] %(message)s', datefmt='%m/%d/%Y %I:%M:%S %p', level=logging.INFO)
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_act', help='num acts', type=int, required=True)
    args = parser.parse_args()

    main(args)



