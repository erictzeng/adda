import logging

import numpy as np
import tensorflow as tf


models = {}

def register_model_fn(name):
    def decorator(fn):
        models[name] = fn
        # set default parameters
        fn.range = None
        fn.mean = None
        fn.bgr = False
        return fn
    return decorator

def get_model_fn(name):
    return models[name]

def preprocessing(inputs, model_fn):
    inputs = tf.cast(inputs, tf.float32)
    channels = inputs.get_shape()[2]
    if channels == 1 and model_fn.num_channels == 3:
        logging.info('Converting grayscale images to RGB')
        inputs = gray2rgb(inputs)
    elif channels == 3 and model_fn.num_channels == 1:
        logging.info('Converting RGB images to grayscale')
        inputs = rgb2gray(inputs)
    if model_fn.range is not None:
        logging.info('Scaling images to range {}.'.format(model_fn.range))
        inputs = model_fn.range * inputs
    if model_fn.default_image_size is not None:
        size = model_fn.default_image_size
        logging.info('Resizing images to [{}, {}]'.format(size, size))
        inputs = tf.image.resize_images(inputs, [size, size])
    if model_fn.mean is not None:
        logging.info('Performing mean subtraction.')
        inputs = inputs - tf.constant(model_fn.mean)
    if model_fn.bgr:
        logging.info('Performing BGR transposition.')
        inputs = inputs[:, :, [2, 1, 0]]
    return inputs

RGB2GRAY = np.array([0.2989, 0.5870, 0.1140], dtype=np.float32)

def rgb2gray(image):
    return tf.reduce_sum(tf.multiply(image, tf.constant(RGB2GRAY)),
                         2,
                         keep_dims=True)

def gray2rgb(image):
    return tf.multiply(image, tf.constant(RGB2GRAY))
