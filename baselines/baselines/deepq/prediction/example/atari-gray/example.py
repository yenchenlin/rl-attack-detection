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
    from tfacvp.util import post_process_gray, pre_process_state_gray

    with tf.Graph().as_default() as graph:
        # Define tensorflow computation graph
        # In this example, I hardcode the arguments num_channel and num_frame for grayscale atari settings
        logging.info('Create model [num_act = %d, num_channel = %d, num_frame = %d] for testing' % (args.num_act, 1, 4))
        model = ActionConditionalVideoPredictionModel(num_act=args.num_act,
                                                    num_channel=1, num_frame=4,
                                                    is_train=False)

        # Get tensorflow session configuration
        config = get_config(args)

        # Load testing state for predicting next frame
        scale = 255.0
        s = np.load(args.data)
        mean = np.load(args.mean)

        with tf.Session(config=config) as sess:
            # Restore the model from checkpoint
            # If you want to combine with your model, you should notice variable scope otherwise you might get some bugs
            logging.info('Loading weights from %s' % (args.load))
            model.restore(sess, args.load)

            # Predict next frame condition on specified action
            logging.info('Predict next frame condition on action %d' % (args.act))

            # To one hot vector
            a = np.identity(args.num_act)[args.act]

            # Predict next frame
            s = pre_process_state_gray(s, mean, (1.0 / scale), 4)
            print np.max(s), np.min(s)
            x_t_1_pred_batch = model.predict(sess, s[np.newaxis, :], a[np.newaxis, :])[0]

            # Post process predicted frame for visualization
            img = x_t_1_pred_batch[0]
            img = post_process_gray(img, mean, scale)
            cv2.imwrite('pred.png' , img)


if __name__ == '__main__':
    logging.basicConfig(format='[%(asctime)s] %(message)s', datefmt='%m/%d/%Y %I:%M:%S %p', level=logging.INFO)
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', help='testing data (.npy), ndarray(shape = [84,84,4])', type=str, required=True)
    parser.add_argument('--mean', help='image mean path (should be shipped with pre-trained model)', type=str, required=True)
    parser.add_argument('--load', help='model weight path (tensorflow checkpoint)', type=str, required=True)
    parser.add_argument('--num_act', help='number of actions in the game\'s action space', type=int, required=True)
    parser.add_argument('--act', help='which action you want to take', type=int, required=True)
    args = parser.parse_args()
    main(args)



