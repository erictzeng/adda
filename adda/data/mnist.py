import gzip
import operator
import os
import struct
from functools import reduce
from urllib.parse import urljoin

import numpy as np

from adda.data import DatasetGroup
from adda.data import ImageDataset
from adda.data import util
from adda.data.dataset import register_dataset



@register_dataset('mnist')
class MNIST(DatasetGroup):
    """The MNIST database of handwritten digits.

    Homepage: http://yann.lecun.com/exdb/mnist/

    Images are 28x28 grayscale images in the range [0, 1].
    """

    base_url = 'http://yann.lecun.com/exdb/mnist/'

    data_files = {
            'train_images': 'train-images-idx3-ubyte.gz',
            'train_labels': 'train-labels-idx1-ubyte.gz',
            'test_images': 't10k-images-idx3-ubyte.gz',
            'test_labels': 't10k-labels-idx1-ubyte.gz',
            }

    num_classes = 10

    def __init__(self, path=None, shuffle=True):
        DatasetGroup.__init__(self, 'mnist', path)
        self.image_shape = (28, 28, 1)
        self.label_shape = ()
        self.shuffle = shuffle
        self._load_datasets()

    def download(self):
        data_dir = self.get_path()
        if not os.path.exists(data_dir):
            os.mkdir(data_dir)
        for filename in self.data_files.values():
            path = self.get_path(filename)
            if not os.path.exists(path):
                url = urljoin(self.base_url, filename)
                util.maybe_download(url, path)

    def _load_datasets(self):
        abspaths = {name: self.get_path(path)
                    for name, path in self.data_files.items()}
        train_images = self._read_images(abspaths['train_images'])
        train_labels = self._read_labels(abspaths['train_labels'])
        test_images = self._read_images(abspaths['test_images'])
        test_labels = self._read_labels(abspaths['test_labels'])
        self.train = ImageDataset(train_images, train_labels,
                                  image_shape=self.image_shape,
                                  label_shape=self.label_shape,
                                  shuffle=self.shuffle)
        self.test = ImageDataset(test_images, test_labels,
                                 image_shape=self.image_shape,
                                 label_shape=self.label_shape,
                                 shuffle=self.shuffle)

    def _read_datafile(self, path, expected_dims):
        """Helper function to read a file in IDX format."""
        base_magic_num = 2048
        with gzip.GzipFile(path) as f:
            magic_num = struct.unpack('>I', f.read(4))[0]
            expected_magic_num = base_magic_num + expected_dims
            if magic_num != expected_magic_num:
                raise ValueError('Incorrect MNIST magic number (expected '
                                 '{}, got {})'
                                 .format(expected_magic_num, magic_num))
            dims = struct.unpack('>' + 'I' * expected_dims,
                                 f.read(4 * expected_dims))
            buf = f.read(reduce(operator.mul, dims))
            data = np.frombuffer(buf, dtype=np.uint8)
            data = data.reshape(*dims)
            return data

    def _read_images(self, path):
        """Read an MNIST image file."""
        return (self._read_datafile(path, 3)
                .astype(np.float32)
                .reshape(-1, 28, 28, 1)
                / 255)

    def _read_labels(self, path):
        """Read an MNIST label file."""
        return self._read_datafile(path, 1)


@register_dataset('mnist2000')
class MNIST2000(MNIST):

    name = 'mnist2000'

    def __init__(self, seed=None, path=None, shuffle=True):
        if seed is None:
            self.seed = hash(self.name) & 0xffffffff
        else:
            self.seed = seed
        MNIST.__init__(self, path=path, shuffle=shuffle)

    def _load_datasets(self):
        abspaths = {name: self.get_path(path)
                    for name, path in self.data_files.items()}
        rand = np.random.RandomState(self.seed)
        train_images = self._read_images(abspaths['train_images'])
        train_labels = self._read_labels(abspaths['train_labels'])
        inds = rand.permutation(len(train_images))[:2000]
        inds.sort()
        train_images = train_images[inds]
        train_labels = train_labels[inds]
        test_images = self._read_images(abspaths['test_images'])
        test_labels = self._read_labels(abspaths['test_labels'])
        self.train = ImageDataset(train_images, train_labels,
                                  image_shape=self.image_shape,
                                  label_shape=self.label_shape,
                                  shuffle=self.shuffle)
        self.test = ImageDataset(test_images, test_labels,
                                 image_shape=self.image_shape,
                                 label_shape=self.label_shape,
                                 shuffle=self.shuffle)
