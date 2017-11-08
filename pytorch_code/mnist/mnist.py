"""Functions for downloading and reading MNIST data."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import gzip
import os
import tempfile

import numpy
from skimage.util import random_noise
from six.moves import urllib
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf

from tensorflow.contrib.learn.python.learn.datasets import base
from tensorflow.python.framework import dtypes
from mnist import *
import time

import numpy
import random
from scipy.ndimage.interpolation import map_coordinates
from scipy.ndimage.filters import gaussian_filter

#SOURCE_URL = 'http://yann.lecun.com/exdb/mnist/'
#This is for sometime LeCun's site is down
SOURCE_URL = 'https://s3.amazonaws.com/lasagne/recipes/datasets/mnist/'

IMAGE_SIZE=28
NUM_LABELS=10
IMAGE_PIXELS=IMAGE_SIZE*IMAGE_SIZE
NUM_CHANNELS=1
PIXEL_DEPTH=255
#SEED=66478 


class DataSet(object):

  def __init__(self,
               images,
               labels,
               fake_data=False,
               one_hot=False,
               dtype=dtypes.float32,
               reshape=True):
    """Construct a DataSet.
    one_hot arg is used only if fake_data is true.  `dtype` can be either
    `uint8` to leave the input as `[0, 255]`, or `float32` to rescale into
    `[0, 1]`.
    """
#    numpy.random.seed(int(time.time()))
    #numpy.random.seed(SEED)
    dtype = dtypes.as_dtype(dtype).base_dtype
    if dtype not in (dtypes.uint8, dtypes.float32):
      raise TypeError('Invalid image dtype %r, expected uint8 or float32' %
                      dtype)
    if fake_data:
      self._num_examples = 10000
      self.one_hot = one_hot
    else:
      assert images.shape[0] == labels.shape[0], (
          'images.shape: %s labels.shape: %s' % (images.shape, labels.shape))
      self._num_examples = images.shape[0]

      # Convert shape from [num examples, rows, columns, depth]
      # to [num examples, rows*columns] (assuming depth == 1)
      if reshape:
        assert images.shape[3] == 1
        #images = images.reshape(images.shape[0],
        #                        images.shape[1] * images.shape[2])
        images = images.reshape((images.shape[0], images.shape[3], images.shape[1], images.shape[2]))
        #images = images.transpose((0, 2, 3, 1))

    self._images = images
    self._labels = labels
    self._epochs_completed = 0
    self._index_in_epoch = 0

    # Shuffle the data
    perm = numpy.arange(self._num_examples)
    numpy.random.shuffle(perm)
    self._images = self._images[perm]
    self._labels = self._labels[perm]

  @property
  def images(self):
    return self._images

  @property
  def labels(self):
    return self._labels

  @property
  def num_examples(self):
    return self._num_examples

  @property
  def epochs_completed(self):
    return self._epochs_completed

  def next_batch(self, batch_size, fake_data=False):
    """Return the next `batch_size` examples from this data set."""
    if fake_data:
      fake_image = [1] * 784
      if self.one_hot:
        fake_label = [1] + [0] * 9
      else:
        fake_label = 0
      return [fake_image for _ in xrange(batch_size)], [
          fake_label for _ in xrange(batch_size)
      ]
    start = self._index_in_epoch
    self._index_in_epoch += batch_size
    if self._index_in_epoch > self._num_examples:
      # Finished epoch
      self._epochs_completed += 1
      # Shuffle the data
      perm = numpy.arange(self._num_examples)
      #numpy.random.seed(SEED)
      numpy.random.shuffle(perm)
      self._images = self._images[perm]
      self._labels = self._labels[perm]
      # Start next epoch
      start = 0
      self._index_in_epoch = batch_size
      assert batch_size <= self._num_examples
    end = self._index_in_epoch

    # Most of the time return the non distorted image
    return self._images[start:end], self._labels[start:end]

def extract_data(filename, num_images):
  """Extract the images into a 4D tensor [image index, y, x, channels].
  Values are rescaled from [0, 255] down to [-0.5, 0.5].
  """
  #print('Extracting', filename)
  with gzip.open(filename) as bytestream:
    bytestream.read(16)
    buf = bytestream.read(IMAGE_SIZE * IMAGE_SIZE * num_images)
    data = numpy.frombuffer(buf, dtype=numpy.uint8).astype(numpy.float32)

    data = (data - (PIXEL_DEPTH / 2.0)) / PIXEL_DEPTH
    data = data.reshape(num_images, IMAGE_SIZE, IMAGE_SIZE, 1)

    return data

def extract_labels(filename, num_images):
  """Extract the labels into a vector of int64 label IDs."""
  #print('Extracting', filename)
  with gzip.open(filename) as bytestream:
    bytestream.read(8)
    buf = bytestream.read(1 * num_images)
    labels = numpy.frombuffer(buf, dtype=numpy.uint8).astype(numpy.int64)
  return labels

def read_data_sets(train_dir,
                   fake_data=False,
                   one_hot=False,
                   dtype=dtypes.float32,
                   reshape=False,
                   validation_size=5000,
                   worker_id=-1,
                   n_workers=-1):
  if fake_data:

    def fake():
      return DataSet([], [], fake_data=True, one_hot=one_hot, dtype=dtype)

    train = fake()
    validation = fake()
    test = fake()
    return base.Datasets(train=train, validation=validation, test=test)

  TRAIN_IMAGES = 'train-images-idx3-ubyte.gz'
  TRAIN_LABELS = 'train-labels-idx1-ubyte.gz'
  TEST_IMAGES = 't10k-images-idx3-ubyte.gz'
  TEST_LABELS = 't10k-labels-idx1-ubyte.gz'

  local_file = base.maybe_download(TRAIN_IMAGES, train_dir,
                                   SOURCE_URL + TRAIN_IMAGES)
  train_images = extract_data(local_file, 60000)

  local_file = base.maybe_download(TRAIN_LABELS, train_dir,
                                   SOURCE_URL + TRAIN_LABELS)
  train_labels = extract_labels(local_file, 60000)

  local_file = base.maybe_download(TEST_IMAGES, train_dir,
                                   SOURCE_URL + TEST_IMAGES)
  test_images = extract_data(local_file, 10000)

  local_file = base.maybe_download(TEST_LABELS, train_dir,
                                   SOURCE_URL + TEST_LABELS)
  test_labels = extract_labels(local_file, 10000)

  if not 0 <= validation_size <= len(train_images):
    raise ValueError(
        'Validation size should be between 0 and {}. Received: {}.'
        .format(len(train_images), validation_size))

  validation_images = test_images
  validation_labels = test_labels
  train_images = train_images
  train_labels = train_labels

  train = DataSet(train_images, train_labels, dtype=dtype, reshape=reshape)
  #train = DataSet(train_images, train_labels_tmp, dtype=dtype, reshape=reshape)

  validation = DataSet(validation_images, validation_labels, dtype=dtype, reshape=reshape)

  return base.Datasets(train=train, validation=validation, test=None)

def load_mnist(train_dir='MNIST-data', worker_id=-1, n_workers=-1):
  return read_data_sets(train_dir, worker_id=worker_id, n_workers=n_workers)

def down_sample(data_set=None, labels=None, down_sample_num=None):
  down_sample_indices = np.random.randint(low=0, high=data_set.shape[0], size=down_sample_num)
  down_samples = np.take(data_set, down_sample_indices, axis=0)
  down_sample_labels = np.take(labels, down_sample_indices)
  return down_samples, down_sample_labels

def add_noise_wrt_distance(batch, crop_shape, padding=None):
  oshape = np.shape(batch[0])
  if padding:
    oshape = (oshape[0] + 2*padding, oshape[1] + 2*padding)
  new_batch = []
  npad = ((padding, padding), (padding, padding), (0, 0))
  var_list = []
  for i in range(len(batch)):
    new_batch.append(batch[i])
    if padding:
      new_batch[i] = np.lib.pad(batch[i], pad_width=npad,
                          mode='constant', constant_values=0)
    nh = random.randint(0, oshape[0] - crop_shape[0])
    nw = random.randint(0, oshape[1] - crop_shape[1])
    new_batch[i] = new_batch[i][nh:nh + crop_shape[0],
                              nw:nw + crop_shape[1]]
    r = np.linalg.norm(np.subtract(batch[i], new_batch[i]))
    var_list.append(r)
  std_var_list = np.array(var_list) / np.linalg.norm(np.array(var_list))
  for i in range(len(std_var_list)):
    std_var_list[i] = std_var_list[i] * 10
  for i in range(len(batch)):
    gaussian_noise = np.random.normal(0, std_var_list[i], (batch[i].shape[0], batch[i].shape[1], batch[i].shape[2]))
    new_batch[i] = new_batch[i] + gaussian_noise
  return np.array(new_batch)

def split_subset_wrt_labels(ori_data, ori_labels):
    split_index_table = [[i/10] for i in range(10)]
    split_data_table = []
    for label_idx, label in enumerate(ori_labels):
        split_index_table[int(label)].append(label_idx)
    for idx in range(10):
       data_per_label = np.take(ori_data, split_index_table[idx][1:], axis=0)
       split_data_table.append(data_per_label)
    return split_data_table

def search_data_in_line(data_point=None, other_label_data=None, num_per_label=1, fraction=0.1):
    sample_index = np.random.randint(low=0, high=other_label_data.shape[0], size=num_per_label)
    sampled_data = np.take(other_label_data, sample_index, axis=0)
    new_data_points = []
    for sampled_data_point in sampled_data:
        # calculate epsilon first
        epsilon = fraction * (np.linalg.norm(data_point)/float(np.linalg.norm(sampled_data_point)))
        #print(np.linalg.norm(data_point)-float(np.linalg.norm(sampled_data_point)))
        # generate new data points
        #print(np.linalg.norm(np.multiply(data_point, 1- epsilon)))
        #print(np.linalg.norm(np.multiply(data_point, 1- epsilon)+np.multiply(sampled_data_point, epsilon)))
        new_data_points.append(np.multiply(data_point, 1- epsilon)+np.multiply(sampled_data_point, epsilon))
#        new_data_points.append(data_point)
    return new_data_points

def line_among_labels(ori_data, ori_labels, num_per_label=1, fraction=0.1):
    label_range = [6, 8]
    split_data_table=split_subset_wrt_labels(ori_data, ori_labels)
    new_train_set = []
    new_train_labels = []
    for dp_idx, data_point in enumerate(ori_data):
        ori_data_label = ori_labels[dp_idx]
        for i in label_range:
            if i == ori_data_label:
                continue
            else:
                new_data_points=search_data_in_line(data_point=data_point, other_label_data=split_data_table[i], num_per_label=num_per_label, fraction=fraction)
                for n_d_p in new_data_points:
                    new_train_set.append(n_d_p)
                    new_train_labels.append(ori_labels[dp_idx])
    return np.array(new_train_set), np.array(new_train_labels)

def add_gaussian_noise(ori_data, mean=0, var=0.01):
    new_batch = []
    for img_idx, image in enumerate(ori_data):
        new_batch.append(random_noise(image, mode='gaussian', mean=mean, var=var))
    return np.array(new_batch)

# with original data batch added into the augmented dataset
'''
def aug_data_set(ori_data, ori_labels, times_expand=1, aug_type="crop"):
    aug_data_list = []
    new_data=ori_data
    new_label=ori_labels
    for time_aug in range(times_expand):
        if aug_type == 'crop':
            crop_data = add_noise_wrt_distance(ori_data, crop_shape=(28, 28), padding=1)
        elif aug_type == 'line_among_labels':
            crop_data, new_train_labels = line_among_labels(ori_data, ori_labels, num_per_label=1, fraction=0.15)
        elif aug_type == "noise":
            crop_data = add_gaussian_noise(ori_data, mean=0, var=0.01)
        elif aug_type == 'fake':
            # this is only used for debug
            crop_data = ori_data
        aug_data_list.append(crop_data)
        new_data = np.concatenate((new_data,aug_data_list[time_aug]),axis=0)
        if aug_type == 'crop' or 'fake' or 'noise':
            new_label = np.concatenate((new_label,ori_labels), axis=0)
        elif aug_type == 'line_among_labels':
            new_label = np.concatenate((new_label,new_train_labels), axis=0)
    return new_data, new_label
'''

# without original data batch added into the augmented dataset
def aug_data_set(ori_data, ori_labels, times_expand=1, aug_type="crop"):
    aug_data_list = []
#    new_data=ori_data
#    new_label=ori_labels
    for time_aug in range(times_expand):
        if aug_type == 'crop':
            crop_data = add_noise_wrt_distance(ori_data, crop_shape=(28, 28), padding=1)
        elif aug_type == 'line_among_labels':
            crop_data, new_train_labels = line_among_labels(ori_data, ori_labels, num_per_label=1, fraction=0.15)
        elif aug_type == "noise":
            crop_data = add_gaussian_noise(ori_data, mean=0, var=0.01)
        elif aug_type == 'fake':
            # this is only used for debug
            crop_data = ori_data
        aug_data_list.append(crop_data)
        if time_aug == 0:
            new_data = aug_data_list[time_aug]
            if aug_type == "crop" or 'fake' or 'noise':
                new_label = ori_labels
            elif aug_type == "line_among_labels":
                new_label = new_train_labels
        else:
            new_data = np.concatenate((new_data,aug_data_list[time_aug]),axis=0)
            if aug_type == 'crop' or 'fake' or 'noise':
                new_label = np.concatenate((new_label,ori_labels), axis=0)
            elif aug_type == 'line_among_labels':
                new_label = np.concatenate((new_label,new_train_labels), axis=0)
    return new_data, new_label

def random_crop(batch, crop_shape, padding=None):
  oshape = np.shape(batch[0])
  if padding:
    oshape = (oshape[0] + 2*padding, oshape[1] + 2*padding)
  new_batch = []
  npad = ((padding, padding), (padding, padding), (0, 0))
  for i in range(len(batch)):
    new_batch.append(batch[i])
    if padding:
      new_batch[i] = np.lib.pad(batch[i], pad_width=npad,
                          mode='constant', constant_values=0)
    nh = random.randint(0, oshape[0] - crop_shape[0])
    nw = random.randint(0, oshape[1] - crop_shape[1])
    new_batch[i] = new_batch[i][nh:nh + crop_shape[0],
                              nw:nw + crop_shape[1]]
  return np.array(new_batch)

def extract_for_binary(train_set=None, train_labels=None, test_set=None, test_labels=None):
  binary_indices_train = []
  binary_indices_test = []
  for i in range(len(train_set)):
    if train_labels[i] == 6 or train_labels[i] == 8:
      binary_indices_train.append(i)
  for i in range(len(test_set)):
    if test_labels[i] == 6 or test_labels[i] == 8:
      binary_indices_test.append(i)
  train_set_binary = np.take(train_set, binary_indices_train, axis=0)
  train_label_binary = np.take(train_labels, binary_indices_train)
  test_set_binary = np.take(test_set, binary_indices_test, axis=0)
  test_label_binary = np.take(test_labels, binary_indices_test)
  return train_set_binary, train_label_binary, test_set_binary, test_label_binary