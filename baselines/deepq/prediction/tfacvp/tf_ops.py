import tensorflow as tf
import numpy as np

def ReLu(x, name, reuse=False):
    with tf.variable_scope(name, reuse=reuse) as scope:
        l = tf.nn.relu(x)
    return l

def Conv2D(x, filter_shape, out_dim, strides, padding, name, reuse=False):
    # x: input tensor (float32)[n, w, h, c]
    # filter_shape: conv2d filter (int)[w, h]
    # out_dim: output channels (int)
    # strides: conv2d stride size (int)
    # padding: padding type (str)
    # name: variable scope (str)
           
    with tf.variable_scope(name, reuse=reuse) as scope:
        in_dim = x.get_shape()[-1]
        w = tf.get_variable('w', shape=filter_shape + [in_dim, out_dim], initializer=tf.contrib.layers.xavier_initializer_conv2d(uniform=True))
        b = tf.get_variable('b', shape=[out_dim], initializer=tf.constant_initializer(0.0))
        l = tf.nn.conv2d(x, w, strides=[1, strides, strides, 1], padding=padding, name='conv2d')
        l = tf.nn.bias_add(l, b, name='bias_add')
    return l

def FC(x, out_dim, name, initializer=tf.contrib.layers.xavier_initializer(uniform=True), reuse=False):
    # x: input tensor (float32)[n, in_dim]
    # out_dim: output channels (int)
    # name: variable scope (str)

    x = tf.contrib.layers.flatten(x)
    with tf.variable_scope(name, reuse=reuse) as scope:
        in_dim = x.get_shape()[-1]
        w = tf.get_variable('w', shape=[in_dim, out_dim], initializer=initializer)
        b = tf.get_variable('b', shape=[out_dim], initializer=tf.constant_initializer(0.0))
        l = tf.add(tf.matmul(x, w), b, name='add')
    return l

def Deconv2D(x, filter_shape, output_shape, out_dim, strides, padding, name, reuse=False):
    # x: input tensor (float32) [n, w, h, c]
    # filter_shape: conv2d filter (int)[w, h]
    # out_dim: output channels (int)
    # strides: conv2d stride size (int)
    # padding: padding type (str)
    # name: variable scope (str)

    with tf.variable_scope(name, reuse=reuse) as scope:
        in_dim = x.get_shape()[-1]
        w = tf.get_variable('w', shape=filter_shape + [out_dim, in_dim], initializer=tf.contrib.layers.xavier_initializer_conv2d(uniform=True))
        b = tf.get_variable('b', shape=[out_dim], initializer=tf.constant_initializer(0.0))
        l = tf.nn.conv2d_transpose(x, w, output_shape=output_shape, strides=[1, strides, strides, 1], padding=padding, name='deconv2d')
        l = tf.nn.bias_add(l, b, name='bias_add')
    return l

