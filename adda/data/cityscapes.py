import tensorflow as tf

from adda.data.dataset import DatasetGroup
from adda.data.dataset import register_dataset


class SegmentationDataset(object):

    def __init__(self, images, labels, shuffle=True):
        self.images = images
        self.labels = labels
        self.shuffle = shuffle

    def __len__(self):
        return len(self.images)

    def tf_ops(self, capacity=32, produce_filenames=False):
        im_path, label_path = tf.train.slice_input_producer(
            [tf.constant(self.images), tf.constant(self.labels)],
            capacity=capacity,
            shuffle=self.shuffle)
        im = tf.read_file(im_path)
        im = tf.image.decode_image(im, channels=3)
        im = tf.cast(im, tf.float32)
        im.set_shape((1024, 2048, 3))
        label = tf.read_file(label_path)
        label = tf.image.decode_image(label, channels=1)
        label = label[:, :, 0]
        label = tf.cast(label, tf.int32)
        label.set_shape((1024, 2048))
        if produce_filenames:
            return im, label, im_path, label_path
        else:
            return im, label


@register_dataset('cityscapes')
class Cityscapes(DatasetGroup):

    num_classes = 19
    ignore_labels = [255]

    def __init__(self, path=None, shuffle=True, download=False,
                 half_crop=False):
        DatasetGroup.__init__(self, 'cityscapes', path=path, download=download)
        self.shuffle = shuffle
        self.half_crop = half_crop
        self._read_datasets()

    def _read_datasets(self):
        with open(self.get_path('train_image_rel.txt'), 'r') as f:
            train_images = list(self.get_path(line.strip()) for line in f)
        with open(self.get_path('train_label_rel.txt'), 'r') as f:
            train_labels = list(self.get_path(line.strip()) for line in f)
        with open(self.get_path('val_image_rel.txt'), 'r') as f:
            val_images = list(self.get_path(line.strip()) for line in f)
        with open(self.get_path('val_label_rel.txt'), 'r') as f:
            val_labels = list(self.get_path(line.strip()) for line in f)
        if self.half_crop:
            self.train = HalfCropDataset(train_images, train_labels,
                                         shuffle=self.shuffle)
            self.val = HalfCropDataset(val_images, val_labels,
                                       shuffle=self.shuffle)
        else:
            self.train = SegmentationDataset(train_images, train_labels,
                                            shuffle=self.shuffle)
            self.val = SegmentationDataset(val_images, val_labels,
                                        shuffle=self.shuffle)


class HalfCropDataset(object):

    def __init__(self, images, labels, shuffle=True):
        self.images = images
        self.labels = labels
        self.shuffle = shuffle
        self.overlap = 210

    def __len__(self):
        return len(self.images)

    def tf_ops(self, capacity=32):
        im_path, label_path = tf.train.slice_input_producer(
            [tf.constant(self.images), tf.constant(self.labels)],
            capacity=capacity,
            shuffle=self.shuffle)
        im_shape = [1024, 1024 + self.overlap, 3]
        label_shape = [1024, 1024 + self.overlap]
        queue = tf.FIFOQueue(capacity, [tf.float32, tf.int32],
                             shapes=[im_shape, label_shape])
        im = tf.read_file(im_path)
        im = tf.image.decode_image(im, channels=3)
        im = tf.cast(im, tf.float32)
        left_im = im[:, :1024 + self.overlap, :]
        right_im = im[:, 1024 - self.overlap:, :]
        left_im.set_shape(im_shape)
        right_im.set_shape(im_shape)
        label = tf.read_file(label_path)
        label = tf.image.decode_image(label, channels=1)
        label = label[:, :, 0]
        label = tf.cast(label, tf.int32)
        label_pad = tf.ones([1024, self.overlap], dtype=tf.int32) * 255
        left_label = tf.concat([label[:, :1024], label_pad], 1)
        right_label = tf.concat([label_pad, label[:, 1024:]], 1)
        left_label.set_shape(label_shape)
        right_label.set_shape(label_shape)
        ims = tf.stack([left_im, right_im], 0)
        labels = tf.stack([left_label, right_label], 0)
        enqueue_op = queue.enqueue_many([ims, labels])
        qr = tf.train.QueueRunner(queue, [enqueue_op])
        tf.train.add_queue_runner(qr)
        return queue.dequeue()

@register_dataset('cityscapes_half_crop')
def CityscapesHalfCrop(*args, **kwargs):
    return Cityscapes(half_crop=True, *args, **kwargs)


if __name__ == '__main__':
    dataset = CityscapesHalfCrop()
    sess = tf.Session()
    im, label = dataset.train.tf_ops()
    tf.train.start_queue_runners(sess)
    print(sess.run(im).shape)
