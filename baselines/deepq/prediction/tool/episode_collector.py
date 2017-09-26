import numpy as np
import tensorflow as tf
import os, sys, cv2

class EpisodeCollector(object):
    '''
        Episode logger, this class is designed to collect state, action for ActionConditionalVideoPrediction training datas
    '''
    def __init__(self, path, preprocess_func, skip=4):
        # path: Where to save .tfrecord file. (str)
        # preprocess_func: Frame preprocess function. (function)
        # skip: Drop $skip frames, since common RL algorithm use 4 frame as one state.
        #       However first 3 frames are black, we don't want to record these state including black frame. (int)
        self.timestep = 0
        self.preprocess_func = preprocess_func
        self.writer = tf.python_io.TFRecordWriter(path)
        self.skip = skip

    def _int64_feature(self, value):
        return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

    def _bytes_feature(self, value):
        return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

    def save(self, s, a, x_next):
        # s: RL state, usually 4 stacked frames (e.g. ndarray, shape=[84, 84, 12])
        # a: action (int)
        # x_next: next frame. (e.g. nadrrray, shape=[84, 84, 3], [210, 160, 3])

        self.timestep += 1
        if self.timestep > self.skip:
            s_raw = s.tostring()

            x_next = self.preprocess_func(x_next)
            x_next_raw = x_next.tostring()

            example = tf.train.Example(features=tf.train.Features(feature={
                'a_t': self._int64_feature(a),
                's_t': self._bytes_feature(s_raw),
                'x_t_1': self._bytes_feature(x_next_raw)}))
            self.writer.write(example.SerializeToString())

    def close(self):
        self.writer.close()
