import logging
import os
import random
from collections import deque
from collections import OrderedDict

import click
import numpy as np
import tensorflow as tf
from tensorflow.contrib import slim
from tqdm import tqdm

import adda


@click.command()
@click.argument('source')
@click.argument('target')
@click.argument('model')
@click.argument('output')
@click.option('--gpu', default='0')
@click.option('--iterations', default=20000)
@click.option('--batch_size', default=50)
@click.option('--display', default=10)
@click.option('--lr', default=1e-4)
@click.option('--stepsize', type=int)
@click.option('--snapshot', default=5000)
@click.option('--weights', required=True)
@click.option('--solver', default='sgd')
@click.option('--adversary', 'adversary_layers', default=[500, 500],
              multiple=True)
@click.option('--adversary_leaky/--adversary_relu', default=True)
@click.option('--seed', type=int)
def main(source, target, model, output,
         gpu, iterations, batch_size, display, lr, stepsize, snapshot, weights,
         solver, adversary_layers, adversary_leaky, seed):
    # miscellaneous setup
    adda.util.config_logging()
    if 'CUDA_VISIBLE_DEVICES' in os.environ:
        logging.info('CUDA_VISIBLE_DEVICES specified, ignoring --gpu flag')
    else:
        os.environ['CUDA_VISIBLE_DEVICES'] = gpu
    logging.info('Using GPU {}'.format(os.environ['CUDA_VISIBLE_DEVICES']))
    if seed is None:
        seed = random.randrange(2 ** 32 - 2)
    logging.info('Using random seed {}'.format(seed))
    random.seed(seed)
    np.random.seed(seed + 1)
    tf.set_random_seed(seed + 2)
    error = False
    try:
        source_dataset_name, source_split_name = source.split(':')
    except ValueError:
        error = True
        logging.error(
            'Unexpected source dataset {} (should be in format dataset:split)'
            .format(source))
    try:
        target_dataset_name, target_split_name = target.split(':')
    except ValueError:
        error = True
        logging.error(
            'Unexpected target dataset {} (should be in format dataset:split)'
            .format(target))
    if error:
        raise click.Abort

    # setup data
    logging.info('Adapting {} -> {}'.format(source, target))
    source_dataset = getattr(adda.data.get_dataset(source_dataset_name),
                             source_split_name)
    target_dataset = getattr(adda.data.get_dataset(target_dataset_name),
                             target_split_name)
    source_im, source_label = source_dataset.tf_ops()
    target_im, target_label = target_dataset.tf_ops()
    model_fn = adda.models.get_model_fn(model)
    source_im = adda.models.preprocessing(source_im, model_fn)
    target_im = adda.models.preprocessing(target_im, model_fn)
    source_im_batch, source_label_batch = tf.train.batch(
        [source_im, source_label], batch_size=batch_size)
    target_im_batch, target_label_batch = tf.train.batch(
        [target_im, target_label], batch_size=batch_size)

    # base network
    source_ft, _ = model_fn(source_im_batch, scope='source')
    target_ft, _ = model_fn(target_im_batch, scope='target')

    # adversarial network
    source_ft = tf.reshape(source_ft, [-1, int(source_ft.get_shape()[-1])])
    target_ft = tf.reshape(target_ft, [-1, int(target_ft.get_shape()[-1])])
    adversary_ft = tf.concat([source_ft, target_ft], 0)
    source_adversary_label = tf.zeros([tf.shape(source_ft)[0]], tf.int32)
    target_adversary_label = tf.ones([tf.shape(target_ft)[0]], tf.int32)
    adversary_label = tf.concat(
        [source_adversary_label, target_adversary_label], 0)
    adversary_logits = adda.adversary.adversarial_discriminator(
        adversary_ft, adversary_layers, leaky=adversary_leaky)

    # losses
    mapping_loss = tf.losses.sparse_softmax_cross_entropy(
        1 - adversary_label, adversary_logits)
    adversary_loss = tf.losses.sparse_softmax_cross_entropy(
        adversary_label, adversary_logits)

    # variable collection
    source_vars = adda.util.collect_vars('source')
    target_vars = adda.util.collect_vars('target')
    adversary_vars = adda.util.collect_vars('adversary')

    # optimizer
    lr_var = tf.Variable(lr, name='learning_rate', trainable=False)
    if solver == 'sgd':
        optimizer = tf.train.MomentumOptimizer(lr_var, 0.99)
    else:
        optimizer = tf.train.AdamOptimizer(lr_var, 0.5)
    mapping_step = optimizer.minimize(
        mapping_loss, var_list=list(target_vars.values()))
    adversary_step = optimizer.minimize(
        adversary_loss, var_list=list(adversary_vars.values()))

    # set up session and initialize
    config = tf.ConfigProto(device_count=dict(GPU=1))
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)
    sess.run(tf.global_variables_initializer())

    # restore weights
    if os.path.isdir(weights):
        weights = tf.train.latest_checkpoint(weights)
    logging.info('Restoring weights from {}:'.format(weights))
    logging.info('    Restoring source model:')
    for src, tgt in source_vars.items():
        logging.info('        {:30} -> {:30}'.format(src, tgt.name))
    source_restorer = tf.train.Saver(var_list=source_vars)
    source_restorer.restore(sess, weights)
    logging.info('    Restoring target model:')
    for src, tgt in target_vars.items():
        logging.info('        {:30} -> {:30}'.format(src, tgt.name))
    target_restorer = tf.train.Saver(var_list=target_vars)
    target_restorer.restore(sess, weights)

    # optimization loop (finally)
    output_dir = os.path.join('snapshot', output)
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    mapping_losses = deque(maxlen=10)
    adversary_losses = deque(maxlen=10)
    bar = tqdm(range(iterations))
    bar.set_description('{} (lr: {:.0e})'.format(output, lr))
    bar.refresh()
    for i in bar:
        mapping_loss_val, adversary_loss_val, _, _ = sess.run(
            [mapping_loss, adversary_loss, mapping_step, adversary_step])
        mapping_losses.append(mapping_loss_val)
        adversary_losses.append(adversary_loss_val)
        if i % display == 0:
            logging.info('{:20} Mapping: {:10.4f}     (avg: {:10.4f})'
                        '    Adversary: {:10.4f}     (avg: {:10.4f})'
                        .format('Iteration {}:'.format(i),
                                mapping_loss_val,
                                np.mean(mapping_losses),
                                adversary_loss_val,
                                np.mean(adversary_losses)))
        if stepsize is not None and (i + 1) % stepsize == 0:
            lr = sess.run(lr_var.assign(lr * 0.1))
            logging.info('Changed learning rate to {:.0e}'.format(lr))
            bar.set_description('{} (lr: {:.0e})'.format(output, lr))
        if (i + 1) % snapshot == 0:
            snapshot_path = target_restorer.save(
                sess, os.path.join(output_dir, output), global_step=i + 1)
            logging.info('Saved snapshot to {}'.format(snapshot_path))

    coord.request_stop()
    coord.join(threads)
    sess.close()


if __name__ == '__main__':
    main()
