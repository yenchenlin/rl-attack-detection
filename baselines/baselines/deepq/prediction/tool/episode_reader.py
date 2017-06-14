import numpy as np
import tensorflow as tf
import os, sys, cv2

class EpisodeReader(object):
    def __init__(self, path, height=84, width=84):
        self.reader = tf.python_io.tf_record_iterator(path=path)
        self.height = height
        self.width = width
    
    def read(self):
        for string_record in self.reader:
            example = tf.train.Example()
            example.ParseFromString(string_record)
            
            a_t = int(example.features.feature['a_t']
                                         .int64_list
                                         .value[0]) 
           
            s_t_string = (example.features.feature['s_t']
                                          .bytes_list
                                          .value[0])

            x_t_1_string = (example.features.feature['x_t_1']
                                          .bytes_list
                                          .value[0])
                       
            s_t_raw = np.fromstring(s_t_string, dtype=np.uint8)
            s_t = s_t_raw.reshape((self.height, self.width, -1))

            x_t_1_raw = np.fromstring(x_t_1_string, dtype=np.uint8)
            x_t_1 = x_t_1_raw.reshape((self.height, self.width, -1))

            s_t = s_t.astype(np.float32)
            x_t_1 = x_t_1.astype(np.float32)
            
            yield s_t, a_t, x_t_1

    def __call__(self):
        yield self.read()


