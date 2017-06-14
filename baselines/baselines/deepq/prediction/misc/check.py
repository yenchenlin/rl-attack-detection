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
        # Create dataset
        logging.info('Create data flow from %s' % args.train)
        train_data = Dataset(directory=args.train, mean_path=args.mean, batch_size=args.batch_size, num_threads=2, capacity=10000)
    
        # Create initializer
        init = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
         
        # Config session
        config = get_config(args)
        
        # Setup summary
        check_summary_writer = tf.summary.FileWriter(os.path.join(args.log, 'check'), graph)

        check_op = tf.cast(train_data()['x_t_1'] * 255.0 + train_data()['mean'], tf.uint8)
 
        tf.summary.image('x_t_1_batch_restore', check_op, collections=['check'])
        check_summary_op = tf.summary.merge_all('check')

        # Start session
        with tf.Session(config=config) as sess:
            coord = tf.train.Coordinator()
            sess.run(init)
            threads = tf.train.start_queue_runners(sess=sess, coord=coord)
            for i in range(10):
                x_t_1_batch, summary = sess.run([check_op, check_summary_op])
                check_summary_writer.add_summary(summary, i)
            coord.request_stop()
            coord.join(threads)
        
    
if __name__ == '__main__':
    logging.basicConfig(format='[%(asctime)s] %(message)s', datefmt='%m/%d/%Y %I:%M:%S %p', level=logging.INFO)
    parser = argparse.ArgumentParser()
    parser.add_argument('--log', help='summary directory', type=str, default='example/log')
    parser.add_argument('--train', help='training data directory', type=str, default='example/train')
    parser.add_argument('--test', help='testing data directory', type=str, default='example/test')
    parser.add_argument('--mean', help='image mean path', type=str, default='example/mean.npy')
    parser.add_argument('--lr', help='learning rate', type=float, default=1e-4)
    parser.add_argument('--epoch', help='epoch', type=int, default=15000000)
    parser.add_argument('--show_per_epoch', help='epoch', type=int, default=1000)
    parser.add_argument('--test_per_epoch', help='epoch', type=int, default=2000)
    parser.add_argument('--batch_size', help='batch size', type=int, default=32)
    parser.add_argument('--test_batch_size', help='batch size', type=int, default=64)
    args = parser.parse_args()

    main(args)



