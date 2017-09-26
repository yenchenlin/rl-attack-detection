import tensorflow as tf
import numpy as np
import cv2
import argparse
import sys, os
import logging

def get_config(args):
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    return config

def get_cv_image(img, mean, scale):
       return img

def main(args):
    from tfacvp.model import ActionConditionalVideoPredictionModel
    from tfacvp.util import post_process_rgb

    with tf.Graph().as_default() as graph:    
        logging.info('Create model [num_act = %d] for testing' % (args.num_act))
        model = ActionConditionalVideoPredictionModel(num_act=args.num_act, is_train=False)
       
        config = get_config(args)
        s = np.load(args.data)
        mean = np.load(args.mean)
        scale = 255.0

        with tf.Session(config=config) as sess:
            logging.info('Loading weights from %s' % (args.load))
            model.restore(sess, args.load)

            for i in range(args.num_act):
                logging.info('Predict next frame condition on action %d' % (i))
                a = np.identity(args.num_act)[i]
                x_t_1_pred_batch = model.predict(sess, s[np.newaxis, :], a[np.newaxis, :])[0]

                img = x_t_1_pred_batch[0]
                img = post_process(img, mean, scale)
                img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                cv2.imwrite('pred-%02d.png' % i, img)
           

if __name__ == '__main__':
    logging.basicConfig(format='[%(asctime)s] %(message)s', datefmt='%m/%d/%Y %I:%M:%S %p', level=logging.INFO)
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', help='testing data npy', type=str, default='example.npy')
    parser.add_argument('--mean', help='image mean path', type=str, default='mean.npy')
    parser.add_argument('--load', help='model weight path', type=str, required=True)
    parser.add_argument('--num_act', help='num acts', type=int, default=9)
    args = parser.parse_args()
    main(args)



