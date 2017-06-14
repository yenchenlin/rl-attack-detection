import tensorflow as tf
import numpy as np
import re

from .tf_ops import ReLu, Conv2D, FC, Deconv2D

NUM_CHANNELS = 3
NUM_FRAMES = 4

class ActionConditionalVideoPredictionModel(object):
    def __init__(self, num_act, inputs=None,
                            is_train=True,
                            with_summary=True,
                            loss_args=None,
                            optimizer_args=None):
        # num_act: number of action in action space (only discrete)
        # inputs: used to create model inputs (dict)
        # is_train: is training phase
        # loss_args: loss function arguments (e.g. lamb)
        # optimizer_args: optimizer arguments (e.g. optimizer type, learning rate, ...) (dict)
        self.is_train = is_train
        self.num_act = num_act
        self.optimizer_args = optimizer_args
        self.loss_args = loss_args
        self._create_input(inputs)
        self._create_model()
        self._create_output()
        self._create_loss()

        if self.is_train:
            self._create_optimizer()
        if with_summary:
            self._create_summary()

    def _create_input(self, inputs):
        # inputs: if None, use tf.placeholder as input
        #         if not None, expected inputs is a dict
        if inputs == None:
            self.inputs = {'s_t': tf.placeholder(dtype=tf.float32, shape=[None, 84, 84, (NUM_CHANNELS * NUM_FRAMES)]),
                       'a_t': tf.placeholder(dtype=tf.int32, shape=[None, self.num_act]),
                       'x_t_1': tf.placeholder(dtype=tf.float32, shape=[None, 84, 84, (NUM_CHANNELS)])}
        else:
            assert type(inputs) is dict
            self.inputs = inputs

    def _create_model(self):
        self.encode = self._create_encoder(self.inputs['s_t'])
        self.act_embed = self._create_action_embedding(self.inputs['a_t'])
        self.decode = self._create_decoder(self.encode, self.act_embed)

    def _create_output(self):
        self.output = self.decode

    def _create_loss(self):
        lamb = self.loss_args['lamb'] if self.loss_args else 0.0
        with tf.variable_scope('loss', reuse=not self.is_train) as scope:
            t = self.inputs['x_t_1']
            penalty = tf.reduce_sum(lamb * tf.stack([tf.nn.l2_loss(var) for var in tf.trainable_variables()]), name='regularization')
            self.loss = tf.reduce_mean(tf.nn.l2_loss(self.output - t, name='l2') + penalty)

    def _create_optimizer(self):
        lr = self.optimizer_args['lr'] if self.optimizer_args else 1e-4
        with tf.variable_scope('optimize', reuse=not self.is_train) as scope:
            # Setup global_step, optimizer
            self.global_step = tf.get_variable('global_step', shape=(), initializer=tf.constant_initializer(0.0), trainable=False)

            self.learning_rate = tf.train.exponential_decay(lr, self.global_step, 1e5, 0.9, staircase=True)
            self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate, name='optimizer')

            # According to original paper code, learning rate of bias is 2x of base learning rate
            grads_vars = self.optimizer.compute_gradients(self.loss)
            bias_pattern = re.compile('.*/b')
            grads_vars_mult = []
            for grad, var in grads_vars:
                if bias_pattern.match(var.op.name):
                    grads_vars_mult.append((grad * 2.0, var))
                else:
                    grads_vars_mult.append((grad, var))

            # According to original paper, gradient should be clipped with [-0.1, 0.1]
            grads_clip = [(tf.clip_by_value(grad, -0.1, 0.1), var) for grad, var in grads_vars_mult]
            self.train = self.optimizer.apply_gradients(grads_clip, global_step=self.global_step)

    def _create_encoder(self, x):
        # x: input image (tensor([batch_size, 84, 84, 12]))
        l = Conv2D(x, [6, 6], 64, 2, 'VALID', 'conv1')
        l = ReLu(l, 'relu1')
        l = Conv2D(l, [6, 6], 64, 2, 'SAME', 'conv2')
        l = ReLu(l, 'relu2')
        l = Conv2D(l, [6, 6], 64, 2, 'SAME', 'conv3')
        l = ReLu(l, 'relu3')
        l = FC(l, 1024, 'ip1')
        l = ReLu(l, 'relu4')
        l = FC(l, 2048, 'enc-factor', initializer=tf.random_uniform_initializer(minval=-1.0, maxval=1.0))
        return l

    def _create_action_embedding(self, act):
        # act: action input (tensor([batch_size, num_act])) (one-hot vector)
        act = tf.cast(act, tf.float32)
        l = FC(act, 2048, 'act-embed', initializer=tf.random_uniform_initializer(minval=-0.1, maxval=0.1))
        return l

    def _create_decoder(self, encode, act_embed):
        # encode: encode layer
        # act_embed: action embedding layer
        batch_size = tf.shape(encode)[0]
        l = tf.multiply(encode, act_embed, name='merge')
        l = FC(l, 1024, 'dec')
        l = FC(l, 64 * 10 * 10, 'ip4')
        l = ReLu(l, 'relu1')
        l = tf.reshape(l, [-1, 10, 10, 64], name='dec-reshape')
        l = Deconv2D(l, [6, 6], [batch_size, 20, 20, 64], 64, 2, 'SAME', 'deconv3')
        l = ReLu(l, 'relu2')
        l = Deconv2D(l, [6, 6], [batch_size, 40, 40, 64], 64, 2, 'SAME', 'deconv2')
        l = ReLu(l, 'relu3')
        l = Deconv2D(l, [6, 6], [batch_size, 84, 84, NUM_CHANNELS], 3, 2, 'VALID', 'x_hat-05')
        return l

    def _create_summary(self):
        if self.is_train:
            tf.summary.scalar("learning_rate", self.learning_rate, collections=['train'])
        tf.summary.scalar("loss", self.loss, collections=['train'])
        tf.summary.image('x_pred_t_1', tf.cast(self.decode * 255.0, tf.uint8), collections=['train'])
        tf.summary.image('x_t_1', tf.cast(self.inputs['x_t_1'] * 255.0, tf.uint8), collections=['train'])

    def restore(self, sess, ckpt, var_scope=None):
        # sess: tf session
        # ckpt: ckpt path (str)
        if var_scope != None:
            all_vars = tf.all_variables()
            g_vars = [k for k in all_vars if k.name.startswith(var_scope)]

        saver = tf.train.Saver({v.op.name[2:]: v for v in g_vars})
        saver.restore(sess, ckpt)

    def predict(self, sess, s, a):
        # sess: tf session
        # s: state at t [batch_size, 84, 84, NUM_CHANNELS * NUM_FRAMES]
        # a: action at t [batch_size, num_act]
        assert s.shape[1:] == (84, 84, NUM_CHANNELS * NUM_FRAMES)
        assert len(a.shape) == 2
        assert a.shape[1] == self.num_act

        return sess.run([self.output], feed_dict={self.inputs['s_t']: s,
                                                  self.inputs['a_t']: a})

