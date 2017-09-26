import cv2
import numpy as np
import tensorflow as tf

def _pre_process(s, mean, scale, num_frame):
    # s: [h, w, c*num_frame]
    # mean: [h, w, c]
    # scale: float32
    #print s.shape, mean.shape, np.tile(mean, [1, 1, 4]).shape
    #s -= np.tile(mean, [1, 1, num_frame])
    s = s - mean
    s = s * scale
    return s

def pre_process_state_rgb(s, mean, scale, num_frame):
    return _pre_process(s, mean, scale, num_frame)

def pre_process_state_gray(s, mean, scale, num_frame):
    s = _transform_state_color_space_np(s)
    mean = _transform_frame_color_space_np(mean)
    return _pre_process(s, mean, scale, num_frame)

def _post_process(x, mean, scale=255.0):
    x *= scale
    x += mean
    x = np.clip(x, 0, scale)
    x = x.astype(np.uint8)
    return x

def post_process_rgb(x, mean, scale):
    return _post_process(x, mean, scale)

def post_process_gray(x, mean, scale):
    # x: [h, w, 1] (assume gray)
    # mean: [h, w, c*num_frame]
    # scale: float32
    mean = _transform_frame_color_space_np(mean)
    return _post_process(x, mean, scale)

def _np_one_hot(x, n):
    y = np.zeros([len(x), n])
    y[np.arange(len(x)), x] = 1
    return y

def _read_and_decode(directory, s_t_shape, num_act, x_t_1_shape):
    filenames = tf.train.match_filenames_once('./%s/*.tfrecords' % (directory))
    filename_queue = tf.train.string_input_producer(filenames)

    reader = tf.TFRecordReader()

    _, serialized_example = reader.read(filename_queue)
    features = tf.parse_single_example(serialized_example,
                                       features={
                                       'a_t': tf.FixedLenFeature([], tf.int64),
                                       's_t' : tf.FixedLenFeature([], tf.string),
                                       'x_t_1' : tf.FixedLenFeature([], tf.string),
                                       })

    s_t = tf.decode_raw(features['s_t'], tf.uint8)
    x_t_1 = tf.decode_raw(features['x_t_1'], tf.uint8)

    s_t = tf.reshape(s_t, s_t_shape)
    x_t_1 = tf.reshape(x_t_1, x_t_1_shape)

    s_t = tf.cast(s_t, tf.float32)
    x_t_1 = tf.cast(x_t_1, tf.float32)

    a_t = tf.cast(features['a_t'], tf.int32)
    a_t = tf.one_hot(a_t, num_act)

    return s_t, a_t, x_t_1

def _transform_frame_color_space(x):
    # x: [h, w, c]
    return tf.image.rgb_to_grayscale(x)

def _transform_state_color_space(s):
    # s: [h, w, c*num_frame]
    num_splits = int(s.shape[-1] / 3)
    return tf.concat([_transform_frame_color_space(x) for x in tf.split(s, num_splits, axis=2)], axis=2)

def _transform_frame_color_space_np(x):
    return cv2.cvtColor(x, cv2.COLOR_RGB2GRAY)[:, :, np.newaxis]

def _transform_state_color_space_np(s):
    # s: [h, w, c*num_frame]
    num_splits = int(s.shape[-1] / 3)
    return np.concatenate([cv2.cvtColor(x, cv2.COLOR_RGB2GRAY)[:,:,np.newaxis] for x in np.split(s, num_splits, axis=2)], axis=2)

