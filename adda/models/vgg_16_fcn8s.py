import os
from contextlib import ExitStack

import click
import numpy as np
import tensorflow as tf
from tensorflow.contrib import slim

from adda.models import register_model_fn


def vgg_arg_scope(weight_decay=0.0005):
  """Defines the VGG arg scope.

  Args:
    weight_decay: The l2 regularization coefficient.

  Returns:
    An arg_scope.
  """
  with slim.arg_scope([slim.conv2d, slim.fully_connected],
                      activation_fn=tf.nn.relu,
                      weights_regularizer=slim.l2_regularizer(weight_decay),
                      biases_initializer=tf.zeros_initializer()):
      with slim.arg_scope([slim.conv2d], padding='SAME') as arg_sc:
          with slim.arg_scope([slim.max_pool2d], padding='SAME') as arg_sc:
              return arg_sc


def upscale(inputs, scale_factor, name=None):
    new_shape = tf.shape(inputs)[1:3] * scale_factor + scale_factor
    out = tf.image.resize_bilinear(inputs, new_shape, name=name)
    return out

def crop(inputs, size, offset, name=None):
    size_shape = tf.shape(size)[1:3]
    h, w = size_shape[0], size_shape[1]
    in_shape = inputs.get_shape()
    b, c = in_shape[0], in_shape[3]
    result = tf.slice(inputs, [0, offset, offset, 0], [-1, h, w, -1], name=name)
    result.set_shape([b, None, None, c])
    return result


@register_model_fn('vgg_16_fcn8s')
def vgg_16_fcn8s(inputs,
                 num_classes=19,
                 is_training=True,
                 dropout_keep_prob=0.5,
                 scope='vgg_16_fcn8s'):
    """Oxford Net VGG 16-Layers version D Example.

    Note: All the fully_connected layers have been transformed to conv2d layers.
          To use in classification mode, resize input to 224x224.

    Args:
      inputs: a tensor of size [batch_size, height, width, channels].
      num_classes: number of predicted classes.
      is_training: whether or not the model is being trained.
      dropout_keep_prob: the probability that activations are kept in the
        dropout layers during training.
      scope: Optional scope for the variables.

    Returns:
      the last op containing the log predictions and end_points dict.
    """
    net = inputs
    with ExitStack() as cm:
        cm.enter_context(slim.arg_scope(vgg_arg_scope()))
        sc = cm.enter_context(tf.variable_scope(scope, 'vgg_16', [inputs]))
        end_points_collection = sc.name + '_end_points'
        # Collect outputs for conv2d, fully_connected and max_pool2d.
        cm.enter_context(slim.arg_scope(
            [slim.conv2d, slim.fully_connected, slim.max_pool2d],
            outputs_collections=end_points_collection))
        #net = slim.repeat(inputs, 2, slim.conv2d, 64, [3, 3], scope='conv1')
        net = tf.pad(net, [[0, 0], [100, 100], [100, 100], [0, 0]])
        net = slim.conv2d(net, 64, 3, padding='VALID', scope='conv1/conv1_1')
        net = slim.conv2d(net, 64, 3, scope='conv1/conv1_2')
        net = slim.max_pool2d(net, [2, 2], scope='pool1')
        net = slim.repeat(net, 2, slim.conv2d, 128, [3, 3], scope='conv2')
        net = slim.max_pool2d(net, [2, 2], scope='pool2')
        net = slim.repeat(net, 3, slim.conv2d, 256, [3, 3], scope='conv3')
        net = pool3 = slim.max_pool2d(net, [2, 2], scope='pool3')
        net = slim.repeat(net, 3, slim.conv2d, 512, [3, 3], scope='conv4')
        net = pool4 = slim.max_pool2d(net, [2, 2], scope='pool4')
        net = slim.repeat(net, 3, slim.conv2d, 512, [3, 3], scope='conv5')
        net = slim.max_pool2d(net, [2, 2], scope='pool5')
        # Use conv2d instead of fully_connected layers.
        net = slim.conv2d(net, 4096, [7, 7], padding='VALID', scope='fc6')
        net = slim.dropout(net, dropout_keep_prob, is_training=is_training,
                           scope='dropout6')
        net = slim.conv2d(net, 4096, [1, 1], scope='fc7')
        net = slim.dropout(net, dropout_keep_prob, is_training=is_training,
                           scope='dropout7')
        net = slim.conv2d(net, num_classes, [1, 1],
                          activation_fn=None,
                          normalizer_fn=None,
                          weights_initializer=tf.zeros_initializer(),
                          scope='fc8')
        upscore2a = upscale(net, 2, name='upscore2a')
        tf.add_to_collection(end_points_collection, upscore2a)
        score_pool4 = slim.conv2d(pool4 * 0.01, 19, 1, activation_fn=None,
                                  weights_initializer=tf.zeros_initializer(),
                                  scope='score_pool4')
        score_pool4c = crop(score_pool4, upscore2a, 5, name='score_pool4c')
        tf.add_to_collection(end_points_collection, score_pool4c)
        fuse_pool4 = tf.add(upscore2a, score_pool4c, name='fuse_pool4')
        tf.add_to_collection(end_points_collection, fuse_pool4)
        upscore_pool4a = upscale(fuse_pool4, 2, name='upscore_pool4a')
        tf.add_to_collection(end_points_collection, upscore_pool4a)
        score_pool3 = slim.conv2d(pool3 * 0.0001, 19, 1, activation_fn=None,
                                  weights_initializer=tf.zeros_initializer(),
                                  scope='score_pool3')
        score_pool3c = crop(score_pool3, upscore_pool4a, 9, name='score_pool3c')
        tf.add_to_collection(end_points_collection, score_pool3c)
        fuse_pool3 = tf.add(upscore_pool4a, score_pool3c, name='fuse_pool3')
        tf.add_to_collection(end_points_collection, fuse_pool3)
        upscore8a = upscale(fuse_pool3, 8, name='upscore8a')
        tf.add_to_collection(end_points_collection, upscore8a)
        net = score = crop(upscore8a, inputs, 31, name='score')
        tf.add_to_collection(end_points_collection, score)
        # Convert end_points_collection into a end_point dict.
        end_points = slim.utils.convert_collection_to_dict(
            end_points_collection)
        return net, end_points
vgg_16_fcn8s.default_image_size = None  # fully convolutional
vgg_16_fcn8s.num_channels = 3
vgg_16_fcn8s.mean = np.array([123.68, 116.779, 103.939], dtype=np.float32)
vgg_16_fcn8s.bgr = False


@click.command()
@click.option('--gpu', default='0')
def main(gpu):
    os.environ['CUDA_VISIBLE_DEVICES'] = gpu
    sess = tf.Session()
    shape = [1, 1024, 2048, 3]
    inputs = tf.placeholder('float', shape)
    labels = tf.placeholder('int32', shape[0:3])
    net, end_points = vgg_16_fcn8s(inputs)
    dummy = np.zeros(shape)
    shape_ops = []
    for tensor in end_points.values():
        shape_ops.append(tf.shape(tensor))
    loss = tf.reduce_sum(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=net, labels=labels))
    step = tf.train.MomentumOptimizer(0.01, 0.9).minimize(loss)
    sess.run(tf.global_variables_initializer())
    shapes = sess.run(shape_ops, feed_dict={inputs: dummy, labels: dummy[:, :, :, 0].astype(int)})
    sess.run(step, feed_dict={inputs: dummy, labels: dummy[:, :, :, 0].astype(int)})
    for end_point, shape in zip(end_points.keys(), shapes):
        print('{:40} {}'.format(end_point, shape))


if __name__ == '__main__':
    main()
