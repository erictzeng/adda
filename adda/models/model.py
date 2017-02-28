import logging

import tensorflow as tf


models = {}

def register_model_fn(name):
    def decorator(fn):
        models[name] = fn
        return fn
    return decorator

def get_model_fn(name):
    return models[name]

def preprocessing(inputs, model_fn):
    inputs = tf.cast(inputs, tf.float32)
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
