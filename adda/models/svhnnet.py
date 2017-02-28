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
                    slim.arg_scope([slim.max_pool2d, slim.conv2d], padding='SAME'))
            net = slim.conv2d(net, 64, 5)
            net = slim.max_pool2d(net, 3, stride=2)
            layers['pool1'] = net
            net = slim.conv2d(net, 64, 5)
            net = slim.max_pool2d(net, 3, stride=2)
            layers['pool2'] = net
            net = slim.conv2d(net, 128, 5)
            layers['conv3'] = net
            net = tf.contrib.layers.flatten(net)
            net = slim.fully_connected(net, 3072)
            layers['fc4'] = net
            net = slim.fully_connected(net, 2048)
            layers['fc5'] = net
            net = slim.fully_connected(net, 10, activation_fn=None)
            layers['fc6'] = net
    if not reuse:
        reuse = True
    return net, layers
svhnnet.default_image_size = 32
svhnnet.mean = np.array([123.68, 116.779, 103.939], dtype=np.float32)
svhnnet.bgr = False
