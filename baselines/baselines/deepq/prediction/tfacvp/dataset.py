import tensorflow as tf
import numpy as np
import logging
import os, glob, cv2, re

from tool.episode_reader import EpisodeReader
from tfacvp.util import _read_and_decode
from tfacvp.util import *

class Dataset(object):
    def __init__(self, directory, num_act, mean_path, num_threads=1, capacity=1e5, batch_size=32,
                scale=(1.0/255.0), s_t_shape=[84, 84, 4], x_t_1_shape=[84, 84, 1], colorspace='gray'):
        self.scale = scale
        self.s_t_shape = s_t_shape
        self.x_t_1_shape = x_t_1_shape

        # Load image mean
        mean = np.load(os.path.join(mean_path))

        # Prepare data flow
        s_t, a_t, x_t_1 = _read_and_decode(directory,
                                        s_t_shape=s_t_shape,
                                        num_act=num_act,
                                        x_t_1_shape=x_t_1_shape)
        self.mean = mean
        self.s_t_batch, self.a_t_batch, self.x_t_1_batch = tf.train.shuffle_batch([s_t, a_t, x_t_1],
                                                            batch_size=batch_size, capacity=capacity,
                                                            min_after_dequeue=int(capacity*0.25),
                                                            num_threads=num_threads)

        # Subtract image mean (according to J Oh design)
        self.mean_const = tf.constant(mean, dtype=tf.float32)
        print(self.mean_const.get_shape())
        self.s_t_batch = (self.s_t_batch - tf.tile(self.mean_const, [1, 1, 4])) * scale
        self.x_t_1_batch = (self.x_t_1_batch - self.mean_const) * scale

    def __call__(self):
        return {'s_t': self.s_t_batch,
                'a_t': self.a_t_batch,
                'x_t_1': self.x_t_1_batch}

class CaffeDataset(object):
    '''
        Used to load data with directory structure in original paper
    '''
    def __init__(self, dir, num_act, mean_path, mode='tf', scale=(1./255.), img_shape=[84, 84], num_frame=4, num_channel=3):
        # dir: image data directory, each image should be named as %05d.png
        # num_act: number of action in action space (only support discrete action)
        # mean_path: mean image file path (NOTE: you must convert mean.binaryproto to npy file)
        # mode: tf or caffe (differ in s, a format)
        # num_frame: initial frame
        # num_channel: number of channel per frame
        self.num_act = num_act
        self.dir = dir
        self.mode = mode
        self.scale = scale
        self.img_shaep = img_shape
        self.num_frame = num_frame
        self.num_channel = num_channel

        pat = re.compile('.*npy')
        if pat.match(mean_path):
            logging.info('Load mean with npy')
            self.mean = np.load(mean_path)
        else:
            import caffe
            logging.info('Load mean with caffe')
            with open(mean_path, 'rb') as mean_file:
                mean_blob = caffe.proto.caffe_pb2.BlobProto()
                mean_bin = mean_file.read()
                mean_blob.ParseFromString(mean_bin)
                self.mean = caffe.io.blobproto_to_array(mean_blob).squeeze()

                if self.mode == 'tf':
                    self.mean = np.transpose(self.mean, [1, 2, 0])

    def _process_frame(self, s, img):
        # s: state np array
        # img: frame input
        img = img.astype(np.float32)
        if self.mode == 'caffe':
            img = np.transpose(img, [2, 0, 1])
        img -= self.mean
        img *= self.scale
        if self.mode == 'tf':
            s[:, :, :-self.num_channel] = s[:, :, self.num_channel:]
            s[:, :, -self.num_channel:] = img
        else:
            s[:-1, :, :, :] = s[1:, :, :, :]
            s[-1, :, :, :] = img
        return s

    def _process_act(self, a, act):
        if self.mode == 'tf':
            a[:-1] = a[1:]
            a[-1] = act
        else:
            a[:, :-1] = a[:, 1:]
            a[:, -1] = act
        return a

    def __call__(self, max_iter=None):
        with open(os.path.join(self.dir, 'act.log')) as act_log:
            cnt_frame = 0
            lim = self.num_frame
            if self.mode == 'tf':
                s = np.zeros(self.img_shape + [self.num_frame * self.num_channel], dtype=np.float32)
                a = np.zeros([self.num_frame, 1], dtype=np.int32)
            else:
                s = np.zeros([self.num_frame, self.num_channel] + self.img_shape, dtype=np.float32)
                a = np.zeros([self.num_frame, 1], dtype=np.int32)

            for filename in sorted(glob.glob(os.path.join(self.dir, '*.png')))[:max_iter]:
                logging.info('%s' % filename)
                img = cv2.imread(filename)

                s = self._process_frame(s, img)
                a = self._process_act(a, int(act_log.readline()[:-1]))

                if cnt_frame < lim:
                    cnt_frame += 1
                else:
                    yield s, _np_one_hot(a[-1], self.num_act)

class NumpyDataset(object):
    def __init__(self, path, mean_path, num_act, scale=(1./255.), s_shape=[84,84,12]):
        # path: tfrecords path
        # num_act: number of action in action space
        # mean_path: mean file path (must be a npy file, with [h, w, c])
        # scale: image scale
        # s_shape: state shape [batch_size, h, w, c * num_frame]
        self.path = path
        self.mean = np.load(mean_path)
        self.num_act = num_act
        self.scale = scale
        self.s_shape = s_shape

    def _preprocess(self, s, a, x_t_1):
        s -= np.tile(self.mean, [4])
        s *= self.scale
        x_t_1 -= self.mean
        x_t_1 *= self.scale
        a = _np_one_hot([a], self.num_act)
        return s, a, x_t_1

    def __call__(self, max_iter=None):
        reader = EpisodeReader(self.path, self.s_shape[0], self.s_shape[1])
        i = 0
        for s, a, x_t_1 in reader.read():
            yield self._preprocess(s, a, x_t_1)
            if max_iter and i >= max_iter:
                break
            i += 1
