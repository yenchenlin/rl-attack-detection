import sys
import caffe
import six
import numpy as np
from collections import OrderedDict
from caffe.proto import caffe_pb2 as PB

import net as N
import cv2
import os

from dataset import CaffeDataset

import argparse
import logging

def post_process(data, mean, scale):
  t = data.copy().squeeze()
  t /= scale
  t += mean
  t = t.clip(0, 255)
  return t.astype('uint8').squeeze().transpose([1, 0, 2]).transpose([0, 2, 1])

class CaffeActionConditionalVideoPredictionModel(object):
    def __init__(self, mean, weight, K, num_act, num_step=1, data_path='test'):
        self.K = K
        self.num_act = num_act
        self.num_step = num_step

        caffe.set_mode_gpu()
        caffe.set_device(0)

        test_net_file, net_proto = N.create_netfile(1, data_path, mean, K, K,
            1, num_act, num_step=self.num_step, mode='test')

        self.test_net = caffe.Net(test_net_file, caffe.TEST)
        self.test_net.copy_from(weight)
    
    def predict(self, s, a, layer='x_hat-05'):
        # s: state (1, 4, 84, 84, 3)
        # a: action (1, 1, num_act)

        '''
        Load data to test_net
        data = [1, K, 84, 84, 3]
        '''
        self.test_net.blobs['data'].data[:] = s
        self.test_net.blobs['act'].data[:] = a
        self.test_net.forward()
        
        pred_data = self.test_net.blobs[args.layer].data[:]

        return pred_data
 
def main(args):
    data = CaffeDataset(dir=args.data, num_act=args.num_act, mean_path=args.mean, mode='caffe')
    model = CaffeActionConditionalVideoPredictionModel(mean=args.mean, weight=args.weight, K=4, num_act=args.num_act)
    
    i = 0
    w = model.test_net.params['conv1'][0].data[:]
    np.save('conv1_w.npy', w)
    for s, a in data(5):
        pred_data = model.predict(s, a)
        print pred_data.shape
        np.save('caffe-%03d.npy' % i, pred_data)
        #pred_img = post_process(pred_data, data.mean, 1./255)
        #cv2.imwrite('%03d-caffe.png' % i, pred_img)
        i += 1
    
if __name__ == '__main__':
    logging.basicConfig(format='[%(asctime)s] %(message)s', datefmt='%m/%d/%Y %I:%M:%S %p', level=logging.INFO)
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', help='testing data directory', type=str, required=True)
    parser.add_argument('--mean', help='image mean path', type=str, required=True)
    parser.add_argument('--weight', help='caffe model', type=str, required=True)
    parser.add_argument('--num_act', help='num acts', type=int, required=True) 
    parser.add_argument('--layer', help='output layer', type=str, required=True)
    args = parser.parse_args()

    main(args)



