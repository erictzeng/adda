from contextlib import ExitStack
from collections import OrderedDict

import numpy as np
import tensorflow as tf
from tensorflow.contrib import slim

from adda.models import register_model_fn


@register_model_fn('svhnnet')
def svhnnet(inputs, scope='svhnnet', is_training=True, reuse=False):
    layers = OrderedDict()
    net = inputs
    with tf.variable_scope(scope, reuse=reuse):
        with ExitStack() as stack:
            stack.enter_context(
                slim.arg_scope(
                    [slim.fully_connected, slim.conv2d],
                    activation_fn=tf.nn.relu,
                    weights_regularizer=slim.l2_regularizer(2.5e-5)))
            stack.enter_context(
                slim.arg_scope([slim.max_pool2d, slim.conv2d],
                               padding='SAME'))
            net = slim.conv2d(net, 64, 5, scope='conv1')
            net = slim.max_pool2d(net, 3, stride=2, scope='pool1')
            layers['pool1'] = net
            net = slim.conv2d(net, 64, 5, scope='conv2')
            net = slim.max_pool2d(net, 3, stride=2, scope='pool2')
            layers['pool2'] = net
            net = slim.conv2d(net, 128, 5, scope='conv3')
            layers['conv3'] = net
            net = tf.contrib.layers.flatten(net)
            net = slim.fully_connected(net, 3072, scope='fc4')
            layers['fc4'] = net
            net = slim.fully_connected(net, 2048, scope='fc5')
            layers['fc5'] = net
            net = slim.fully_connected(net, 10, activation_fn=None, scope='fc6')
            layers['fc6'] = net
    return net, layers
svhnnet.default_image_size = 32
svhnnet.num_channels = 1
svhnnet.range = 255
svhnnet.mean = np.array([123.68, 116.779, 103.939], dtype=np.float32)
svhnnet.bgr = False
