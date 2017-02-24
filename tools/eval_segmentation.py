import logging
import os
import time
from collections import OrderedDict

import click
import numpy as np
import tensorflow as tf
from tqdm import tqdm

import adda


def preprocessing(inputs, model_fn):
    inputs = tf.cast(inputs, tf.float32)
    if model_fn.default_image_size is not None:
        size = model_fn.default_image_size
        inputs = tf.image.resize_images(inputs, [size, size])
    if model_fn.mean is not None:
        inputs = inputs - tf.constant(model_fn.mean)
    if model_fn.bgr:
        inputs = inputs[:, :, [2, 1, 0]]
    return inputs

def remove_first_scope(name):
    return '/'.join(name.split('/')[1:])

def collect_vars(scope, start=None, end=None, prepend_scope=None):
    vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=scope)
    var_dict = OrderedDict()
    if isinstance(start, str):
        for i, var in enumerate(vars):
            var_name = remove_first_scope(var.op.name)
            if var_name.startswith(start):
                start = i
                break
    if isinstance(end, str):
        for i, var in enumerate(vars):
            var_name = remove_first_scope(var.op.name)
            if var_name.startswith(end):
                end = i
                break
    for var in vars[start:end]:
        var_name = remove_first_scope(var.op.name)
        if prepend_scope is not None:
            var_name = os.path.join(prepend_scope, var_name)
        var_dict[var_name] = var
    return var_dict


def count_intersection_and_union(predictions, gt, num_classes, ignore=[]):
    predictions = predictions.copy()
    for ignore_label in ignore:
        predictions[gt == ignore_label] = ignore_label
    intersections = np.zeros((num_classes,))
    unions = np.zeros((num_classes,))
    for label in range(num_classes):
        if label in ignore:
            continue
        pred_map = predictions == label
        gt_map = gt == label
        intersections[label] = np.sum(pred_map & gt_map)
        unions[label] = np.sum(pred_map | gt_map)
    return intersections, unions

def iou_str(iou):
    result = []
    for val in iou:
        result.append('{:4.2f}'.format(val))
    return '  '.join(result)


@click.command()
@click.argument('dataset')
@click.argument('split')
@click.argument('model')
@click.argument('weights')
@click.option('--gpu', default='0')
def main(dataset, split, model, weights, gpu):
    adda.util.config_logging()
    if 'CUDA_VISIBLE_DEVICES' in os.environ:
        logging.info('CUDA_VISIBLE_DEVICES specified, ignoring --gpu flag')
    else:
        os.environ['CUDA_VISIBLE_DEVICES'] = gpu
        logging.info('Using GPU {}'.format(gpu))

    dataset_name = dataset
    split_name = split
    dataset = adda.data.get_dataset(dataset, shuffle=False)
    split = getattr(dataset, split)
    model_fn = adda.models.get_model_fn(model)
    im, label = split.tf_ops(capacity=2)
    im = preprocessing(im, model_fn)
    im_batch, label_batch = tf.train.batch([im, label], num_threads=4, batch_size=1)

    net, layers = model_fn(im_batch, is_training=False)
    net = tf.argmax(net, 3)

    config = tf.ConfigProto(device_count=dict(GPU=1))
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)
    sess.run(tf.global_variables_initializer())
    var_dict = collect_vars(model)
    restorer = tf.train.Saver(var_list=var_dict)
    if os.path.isdir(weights):
        weights = tf.train.latest_checkpoint(weights)
    logging.info('Evaluating {}'.format(weights))
    restorer.restore(sess, weights)

    intersections = np.zeros((dataset.num_classes,))
    unions = np.zeros((dataset.num_classes,))
    for i in tqdm(range(len(split))):
        start = time.time()
        predictions, im, gt = sess.run([net, im_batch, label_batch])
        forward_time = time.time() - start
        start = time.time()
        im_intersection, im_union = count_intersection_and_union(
            predictions[0], gt[0], dataset.num_classes,
            ignore=dataset.ignore_labels)
        iou_time = time.time() - start
        intersections += im_intersection
        unions += im_union
        logging.info('Image {}: forward: {:.4f} seconds,\tiou: {:.4f} seconds'
                     .format(i, forward_time, iou_time))
        ious = intersections / unions
        miou = np.mean(ious)
        logging.info('        IoU so far: {}    AVG: {:.2f}'
                     .format(iou_str(ious), miou))
    ious = intersections / unions
    print(ious)
    print(np.mean(ious))
    

if __name__ == '__main__':
    main()
