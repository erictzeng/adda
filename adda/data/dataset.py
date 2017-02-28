import os

import numpy as np
import tensorflow as tf
from tensorflow.contrib.learn.python.learn.dataframe.queues.feeding_queue_runner import FeedingQueueRunner


class DatasetGroup(object):

    def __init__(self, name, path=None, download=True):
        self.name = name
        if path is None:
            path = os.path.join(os.getcwd(), 'data')
        self.path = path
        if download:
            self.download()

    def get_path(self, *args):
        return os.path.join(self.path, self.name, *args)

    def download(self):
        """Download the dataset(s).

        This method only performs the download if necessary. If the dataset
        already resides on disk, it is a no-op.
        """
        pass


class ImageDataset(object):

    def __init__(self, images, labels, image_shape=None, label_shape=None,
                 shuffle=True):
        self.images = images
        self.labels = labels
        self.image_shape = image_shape
        self.label_shape = label_shape
        self.shuffle = shuffle

    def __len__(self):
        return len(self.images)

    def __iter__(self):
        inds = np.arange(len(self))
        if self.shuffle:
            np.random.shuffle(inds)
        for ind in inds:
            yield self.images[ind], self.labels[ind]

    def feed(self, im, label, epochs=None):
        epochs_elapsed = 0
        while epochs is None or epochs_elapsed < epochs:
            for entry in self:
                yield {im: entry[0], label: entry[1]}
            epochs_elapsed += 1

    def tf_ops(self, capacity=32):
        im = tf.placeholder(tf.float32, shape=self.image_shape)
        label = tf.placeholder(tf.int32, shape=self.label_shape)
        if self.image_shape is None or self.label_shape is None:
            shapes = None
        else:
            shapes = [self.image_shape, self.label_shape]
        queue = tf.FIFOQueue(capacity, [tf.float32, tf.int32], shapes=shapes)
        enqueue_op = queue.enqueue([im, label])
        fqr = FeedingQueueRunner(queue, [enqueue_op],
                                 feed_fns=[self.feed(im, label).__next__])
        tf.train.add_queue_runner(fqr)
        return queue.dequeue()


class FilenameDataset(object):

    def tf_ops(self, capacity=32):
        im, label = tf.train.slice_input_producer(
            [tf.constant(self.images), tf.constant(self.labels)],
            capacity=capacity,
            shuffle=True)
        im = tf.read_file(im)
        im = tf.image.decode_image(im, channels=3)
        return im, label


datasets = {}


def register_dataset(name):
    def decorator(cls):
        datasets[name] = cls
        return cls
    return decorator


def get_dataset(name, *args, **kwargs):
    return datasets[name](*args, **kwargs)
