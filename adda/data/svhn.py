import os
from urllib.parse import urljoin

import numpy as np
from scipy.io import loadmat

from adda.data import DatasetGroup
from adda.data import ImageDataset
from adda.data import util
from adda.data.dataset import register_dataset


@register_dataset('svhn')
class SVHN(DatasetGroup):
    """The Street View House Numbers Dataset.

    This DatasetGroup corresponds to format 2, which consists of center-cropped
    digits.

    Homepage: http://ufldl.stanford.edu/housenumbers/

    Images are 32x32 RGB images in the range [0, 1].
    """

    base_url = 'http://ufldl.stanford.edu/housenumbers/'

    data_files = {
            'train': 'train_32x32.mat',
            'test': 'test_32x32.mat',
            #'extra': 'extra_32x32.mat',
            }

    def __init__(self, path=None, shuffle=True):
        DatasetGroup.__init__(self, 'svhn', path=path)
        self.train_on_extra = False  # disabled
        self.image_shape = (32, 32, 3)
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
        train_mat = loadmat(abspaths['train'])
        train_images = train_mat['X'].transpose((3, 0, 1, 2))
        train_labels = train_mat['y'].squeeze()
        if self.train_on_extra:
            extra_mat = loadmat(abspaths['extra'])
            train_images = np.vstack((train_images,
                                      extra_mat['X'].transpose((3, 0, 1, 2))))
            train_labels = np.concatenate((train_labels,
                                           extra_mat['y'].squeeze()))
        train_labels[train_labels == 10] = 0
        train_images = train_images.astype(np.float32) / 255
        test_mat = loadmat(abspaths['test'])
        test_images = test_mat['X'].transpose((3, 0, 1, 2))
        test_images = test_images.astype(np.float32) / 255
        test_labels = test_mat['y'].squeeze()
        test_labels[test_labels == 10] = 0
        self.train = ImageDataset(train_images, train_labels,
                                  image_shape=self.image_shape,
                                  label_shape=self.label_shape,
                                  shuffle=self.shuffle)
        self.test = ImageDataset(test_images, test_labels,
                                  image_shape=self.image_shape,
                                  label_shape=self.label_shape,
                                  shuffle=self.shuffle)
