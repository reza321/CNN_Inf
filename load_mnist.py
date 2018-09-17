# Adapted from https://github.com/tensorflow/tensorflow/blob/master/tensorflow/contrib/learn/python/learn/datasets/mnist.py

import gzip
import numpy as np
import pickle
from tensorflow.contrib.learn.python.learn.datasets import base
from dataset import DataSet


def _read32(bytestream):
  dt = np.dtype(np.uint32).newbyteorder('>')
  return np.frombuffer(bytestream.read(4), dtype=dt)[0]


def extract_images(f):
  """Extract the images into a 4D uint8 np array [index, y, x, depth].
  Args:
    f: A file object that can be passed into a gzip reader.
  Returns:
    data: A 4D unit8 np array [index, y, x, depth].
  Raises:
    ValueError: If the bytestream does not start with 2051.
  """
  print('Extracting', f.name)
  with gzip.GzipFile(fileobj=f) as bytestream:
    magic = _read32(bytestream)
    if magic != 2051:
      raise ValueError('Invalid magic number %d in MNIST image file: %s' %
                       (magic, f.name))
    num_images = _read32(bytestream)
    rows = _read32(bytestream)
    cols = _read32(bytestream)
    buf = bytestream.read(rows * cols * num_images)
    data = np.frombuffer(buf, dtype=np.uint8)
    data = data.reshape(num_images, rows, cols, 1)
    return data


def extract_labels(f, one_hot=False, num_classes=10):
  """Extract the labels into a 1D uint8 np array [index].
  Args:
    f: A file object that can be passed into a gzip reader.
    one_hot: Does one hot encoding for the result.
    num_classes: Number of classes for the one hot encoding.
  Returns:
    labels: a 1D unit8 np array.
  Raises:
    ValueError: If the bystream doesn't start with 2049.
  """
  print('Extracting', f.name)
  with gzip.GzipFile(fileobj=f) as bytestream:
    magic = _read32(bytestream)
    if magic != 2049:
      raise ValueError('Invalid magic number %d in MNIST label file: %s' %
                       (magic, f.name))
    num_items = _read32(bytestream)
    buf = bytestream.read(num_items)
    labels = np.frombuffer(buf, dtype=np.uint8)
    if one_hot:
      return dense_to_one_hot(labels, num_classes)
    return labels


def load_mnist(train_dir, validation_size=5000):

  SOURCE_URL = 'http://yann.lecun.com/exdb/mnist/'
 
  TRAIN_IMAGES = 'train-images-idx3-ubyte.gz'
  TRAIN_LABELS = 'train-labels-idx1-ubyte.gz'
  TEST_IMAGES = 't10k-images-idx3-ubyte.gz'
  TEST_LABELS = 't10k-labels-idx1-ubyte.gz'

  local_file = base.maybe_download(TRAIN_IMAGES, train_dir,
                                   SOURCE_URL + TRAIN_IMAGES)
  with open(local_file, 'rb') as f:
    train_images = extract_images(f)

  local_file = base.maybe_download(TRAIN_LABELS, train_dir,
                                   SOURCE_URL + TRAIN_LABELS)
  with open(local_file, 'rb') as f:
    train_labels = extract_labels(f)

  local_file = base.maybe_download(TEST_IMAGES, train_dir,
                                   SOURCE_URL + TEST_IMAGES)
  with open(local_file, 'rb') as f:
    test_images = extract_images(f)

  local_file = base.maybe_download(TEST_LABELS, train_dir,
                                   SOURCE_URL + TEST_LABELS)
  with open(local_file, 'rb') as f:
    test_labels = extract_labels(f)

  if not 0 <= validation_size <= len(train_images):
    raise ValueError(
        'Validation size should be between 0 and {}. Received: {}.'
        .format(len(train_images), validation_size))

  validation_images = train_images[:validation_size]
  validation_labels = train_labels[:validation_size]
  train_images = train_images[validation_size:]
  train_labels = train_labels[validation_size:]

  train_images = train_images.astype(np.float32) / 255
  validation_images = validation_images.astype(np.float32) / 255
  test_images = test_images.astype(np.float32) / 255

  train = DataSet(train_images, train_labels)
  validation = DataSet(validation_images, validation_labels)
  test = DataSet(test_images, test_labels)

  return base.Datasets(train=train, validation=validation, test=test)

def load_inv_size_mnist(train_dir, size_divisor,validation_size=5000, random_seed=0):
  np.random.seed(random_seed)
  data_sets = load_mnist(train_dir, validation_size)

  train_images = data_sets.train.x
  train_labels = data_sets.train.labels
  perm = np.arange(len(train_labels))
  np.random.shuffle(perm)
  num_to_keep = int(len(train_labels) /size_divisor)
  perm = perm[:num_to_keep]
  train_images = train_images[perm, :]
  train_labels = train_labels[perm]

  validation_images = data_sets.validation.x
  validation_labels = data_sets.validation.labels
  # perm = np.arange(len(validation_labels))
  # np.random.shuffle(perm)
  # num_to_keep = int(len(validation_labels) / 10)
  # perm = perm[:num_to_keep]  
  # validation_images = validation_images[perm, :]
  # validation_labels = validation_labels[perm]

  test_images = data_sets.test.x
  test_labels = data_sets.test.labels
  # perm = np.arange(len(test_labels))
  # np.random.shuffle(perm)
  # num_to_keep = int(len(test_labels) / 10)
  # perm = perm[:num_to_keep]
  # test_images = test_images[perm, :]
  # test_labels = test_labels[perm]

  train = DataSet(train_images, train_labels)
  validation = DataSet(validation_images, validation_labels)
  test = DataSet(test_images, test_labels)

  return base.Datasets(train=train, validation=validation, test=test)

def load_inv_size_label_mnist(train_dir, size_divisor,labels_list,validation_size=5000, random_seed=0):
  
  np.random.seed(random_seed)
  data_sets = load_mnist(train_dir, validation_size)


  train_images = data_sets.train.x
  train_labels = data_sets.train.labels
  perm = np.arange(len(train_labels))
  np.random.shuffle(perm)
  num_to_keep = int(len(train_labels) /size_divisor)
  perm = perm[:num_to_keep]


  train_images = data_sets.train.x[perm, :]
  train_labels = data_sets.train.labels[perm]

  validation_images = data_sets.validation.x
  validation_labels = data_sets.validation.labels

  test_images = data_sets.test.x
  test_labels = data_sets.test.labels

  tarin_idx_of_labels = [i for i in range(len(train_labels)) if train_labels[i] in labels_list]
  train_images=train_images[tarin_idx_of_labels]
  train_labels=train_labels[tarin_idx_of_labels]

  validation_idx_of_labels = [i for i in range(len(validation_labels)) if validation_labels[i] in labels_list]
  validation_images=validation_images[validation_idx_of_labels]
  validation_labels=validation_labels[validation_idx_of_labels]

  test_idx_of_labels = [i for i in range(len(test_labels)) if test_labels[i] in labels_list]
  test_images=test_images[test_idx_of_labels]
  test_labels=test_labels[test_idx_of_labels]


  train = DataSet(train_images, train_labels)
  validation = DataSet(validation_images, validation_labels)
  test = DataSet(test_images, test_labels)

  return base.Datasets(train=train, validation=validation, test=test)  

def load_firstfold_GPS_inv_size(train_dir, size_divisor,validation_size=5000, random_seed=0):
  np.random.seed(random_seed)

  # data_sets = load_mnist(train_dir, validation_size)
  # Train_X=np.array(kfold_dataset[0][0]).reshape(len(kfold_dataset[0][0])) # len ---> 19287 --> shape:(19287,992)
  # Train_labels=np.array(kfold_dataset[0][1]).reshape(len(kfold_dataset[0][1]),) # len ---> 19287: (0,1,2,3,4) labels
  # Test_X=np.array(kfold_dataset[0][2]).reshape([len(kfold_dataset[0][2]),-1])    # len ---> 4822 ---> shape: (4822,992)
  # Test_labels=np.array(kfold_dataset[0][4]).reshape(len(kfold_dataset[0][4]),)  # len ---> 4822 (0,1,2,3,4)
  
  Train_X=np.array(kfold_dataset[0][0])# len ---> 19287 --> shape:(19287,1,284,4)
  Train_labels=np.array(kfold_dataset[0][1]) # len ---> 19287: (0,1,2,3,4) labels
  Test_X=np.array(kfold_dataset[0][2])    # len ---> 4822 ---> shape: (4822,1,248,4)
  Test_labels=np.array(kfold_dataset[0][4]) # len ---> 4822 (0,1,2,3,4)  

  train_images = Train_X
  train_labels = Train_labels

  perm = np.arange(len(train_labels))
  np.random.shuffle(perm)
  num_to_keep = int(len(train_labels) /size_divisor)
  perm = perm[:num_to_keep]
  train_images = train_images[perm, :]
  train_labels = train_labels[perm]

  validation_images = Train_X       #-------> It is the same as training phase, so i have to change it later 
  validation_labels = Train_labels  #-------> It is the same as training phase, so i have to change it later

  test_images = Test_X
  test_labels = Test_labels

  train = DataSet(train_images, train_labels)
  validation = DataSet(validation_images, validation_labels)
  test = DataSet(test_images, test_labels)

  return base.Datasets(train=train, validation=validation, test=test)

def load_firstfold_GPS_inv_size2(Train_X,Train_Y,Val_X,Val_Y,Test_X,Test_Y,train_dir,validation_size=5000, random_seed=0):

  train = DataSet(Train_X, Train_Y)
  validation = DataSet(Val_X,Val_Y)
  test = DataSet(Test_X, Test_Y)

  return base.Datasets(train=train, validation=validation, test=test)






