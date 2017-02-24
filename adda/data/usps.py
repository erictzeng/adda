import gzip
import os
from urllib.parse import urljoin

import numpy as np

from adda.data import DatasetGroup
from adda.data import ImageDataset
from adda.data import util
from adda.data.dataset import register_dataset


@register_dataset('usps')
class USPS(DatasetGroup):
    """USPS handwritten digits.

    Homepage: http://statweb.stanford.edu/~tibs/ElemStatLearn/data.html

    Images are 16x16 grayscale images in the range [0, 1].
    """

    base_url = 'http://statweb.stanford.edu/~tibs/ElemStatLearn/datasets/'

    data_files = {
        'train': 'zip.train.gz',
        'test': 'zip.test.gz'
        }

    def __init__(self, path=None, download=True):
        DatasetGroup.__init__(self, 'usps', path=path, download=download)
        self.image_shape = (16, 16, 1)
        self.label_shape = ()
        self._load_datasets()

    def download(self):
        for filename in self.data_files.values():
            path = self.get_path(filename)
            if not os.path.exists(path):
                url = urljoin(self.base_url, filename)
                util.maybe_download(url, path)

    def _load_datasets(self):
        abspaths = {name: self.get_path(path)
                    for name, path in self.data_files.items()}
        train_images, train_labels = self._read_datafile(abspaths['train'])
        test_images, test_labels = self._read_datafile(abspaths['test'])
        self.train = ImageDataset(train_images, train_labels,
                                  image_shape=self.image_shape,
                                  label_shape=self.label_shape)
        self.test = ImageDataset(test_images, test_labels,
                                  image_shape=self.image_shape,
                                  label_shape=self.label_shape)

    def _read_datafile(self, path):
        """Read the proprietary USPS digits data file."""
        labels, images = [], []
        with gzip.GzipFile(path) as f:
            for line in f:
                vals = line.strip().split()
                labels.append(float(vals[0]))
                images.append([float(val) for val in vals[1:]])
        labels = np.array(labels, dtype=np.float32)
        images = np.array(images, dtype=np.float32).reshape(-1, 16, 16, 1)
        images = (images + 1) / 2
        return images, labels
