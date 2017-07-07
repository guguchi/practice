#coding:utf-8

import os
import sys
import tarfile
from six.moves import urllib
import tensorflow as tf


def maybe_download_and_extract(data_dir):
    """データダウンロード"""
    dest_directory = data_dir
    if not os.path.exists(dest_directory):
        os.makedirs(dest_directory)
    DATA_URL = 'http://www.cs.toronto.edu/~kriz/cifar-10-binary.tar.gz'
    filename = DATA_URL.split('/')[-1]
    filepath = os.path.join(dest_directory, filename)
    if not os.path.exists(filepath):
        def _progress(count, block_size, total_size):
            sys.stdout.write('\r>> Downloading %s %.1f%%' % (filename,
                float(count * block_size) / float(total_size) * 100.0))
            sys.stdout.flush()
        filepath, _ = urllib.request.urlretrieve(DATA_URL, filepath, _progress)
        print()
        statinfo = os.stat(filepath)
        print('Successfully downloaded', filename, statinfo.st_size, 'bytes.')
    extracted_dir_path = os.path.join(dest_directory, 'cifar-10-batches-bin')
    if not os.path.exists(extracted_dir_path):
        tarfile.open(filepath, 'r:gz').extractall(dest_directory)


def read_cifar10(filename_queue):
    """Reads and parses examples from CIFAR10 data files."""

    class CIFAR10Record(object):
        pass
    result = CIFAR10Record()

    label_bytes = 1
    result.height = 32
    result.width = 32
    result.depth = 3
    image_bytes = result.height * result.width * result.depth
    record_bytes = label_bytes + image_bytes

    reader = tf.FixedLengthRecordReader(record_bytes=record_bytes)

    ##readerにqueueを渡してファイルを開く
    result.key, value = reader.read(filename_queue)

    # 6. readの結果からデータをdecode
    record_bytes = tf.decode_raw(value, tf.uint8)

    result.label = tf.cast(
        tf.strided_slice(record_bytes, [0], [label_bytes]), tf.int32)

    depth_major = tf.reshape(
        tf.strided_slice(record_bytes, [label_bytes],
                        [label_bytes + image_bytes]),
                        [result.depth, result.height, result.width])
    result.uint8image = tf.transpose(depth_major, [1, 2, 0])

    return result


def _generate_image_and_label_batch(image, label, min_queue_examples,
                                    batch_size, shuffle):
    """Construct a queued batch of images and labels.
    Args:
        image: 3-D Tensor of [height, width, 3] of type.float32.
        label: 1-D Tensor of type.int32
        min_queue_examples: int32, minimum number of samples to retain
        in the queue that provides of batches of examples.
        batch_size: Number of images per batch.
        shuffle: boolean indicating whether to use a shuffling queue.
    Returns:
        images: Images. 4D tensor of [batch_size, height, width, 3] size.
        labels: Labels. 1D tensor of [batch_size] size.
    """
    # Create a queue that shuffles the examples, and then
    # read 'batch_size' images + labels from the example queue.
    num_preprocess_threads = 16
    if shuffle:
        images, label_batch = tf.train.shuffle_batch(
            [image, label],
            batch_size=batch_size,
            num_threads=num_preprocess_threads,
            capacity=min_queue_examples + 1 * batch_size,
            min_after_dequeue=min_queue_examples)
    else:
        images, label_batch = tf.train.batch(
            [image, label],
            batch_size=batch_size,
            num_threads=num_preprocess_threads,
            capacity=min_queue_examples + 1 * batch_size)

    label = tf.reshape(label_batch, [batch_size])
    one_hot_label = tf.one_hot(label, depth = 10)

    images = tf.reshape(images, [batch_size, 32*32])

    return images, one_hot_label


def train_input(data_dir, batch_size):
    filenames = [os.path.join(data_dir, 'data_batch_%d.bin' % i)
                 for i in xrange(1, 6)]
    for f in filenames:
      if not tf.gfile.Exists(f):
        raise ValueError('Failed to find file: ' + f)

    # Create a queue that produces the filenames to read.
    filename_queue = tf.train.string_input_producer(filenames)

    # Read examples from files in the filename queue.
    read_input = read_cifar10(filename_queue)
    gray_image = tf.image.rgb_to_grayscale(read_input.uint8image)
    gray_image = tf.cast(gray_image, tf.float32)
    float_image = tf.image.per_image_standardization(gray_image)

    float_image.set_shape([32, 32, 1])
    read_input.label.set_shape([1])

    min_fraction_of_examples_in_queue = 0.4
    NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN = 50000
    min_queue_examples = int(NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN *
                            min_fraction_of_examples_in_queue)
    print ('Filling queue with %d CIFAR images before starting to train. '
            'This will take a few minutes.' % min_queue_examples)

    return _generate_image_and_label_batch(float_image, read_input.label,
                                           min_queue_examples, batch_size,
                                           shuffle=True)


def test_input(data_dir, batch_size):
    filenames = [os.path.join(data_dir, 'test_batch.bin')]

    for f in filenames:
        if not tf.gfile.Exists(f):
            raise ValueError('Failed to find file: ' + f)

    # Create a queue that produces the filenames to read.
    filename_queue = tf.train.string_input_producer(filenames)

    # Read examples from files in the filename queue.
    read_input = read_cifar10(filename_queue)
    gray_image = tf.image.rgb_to_grayscale(read_input.uint8image)
    gray_image = tf.cast(gray_image, tf.float32)
    float_image = tf.image.per_image_standardization(gray_image)

    float_image.set_shape([32, 32, 1])
    read_input.label.set_shape([1])

    # Ensure that the random shuffling has good mixing properties.
    min_fraction_of_examples_in_queue = 0.4
    NUM_EXAMPLES_PER_EPOCH_FOR_EVAL = 10000
    min_queue_examples = int(NUM_EXAMPLES_PER_EPOCH_FOR_EVAL *
                           min_fraction_of_examples_in_queue)

    # Generate a batch of images and labels by building up a queue of examples.
    # shuffle : False
    return _generate_image_and_label_batch(float_image, read_input.label,
                                            min_queue_examples, batch_size,
                                            shuffle=True)

"""
if __name__ == '__main__':

    data_dir = './'
    batch_size = 32
    max_steps = 10
    maybe_download_and_extract(data_dir)

    images, labels = train_input(data_dir + 'cifar-10-batches-bin/', batch_size)

    # 8. 実行
    sess = tf.Session()
    tf.train.start_queue_runners(sess=sess)
    for step in xrange(max_steps):
        img, label = sess.run([images, labels])
        print(label)
    print("FIN.")
    img = np.reshape(img, [batch_size, 32, 32])
    print img[0]
    #plt.gray()
    plt.subplot(151)
    plt.imshow(img[0], interpolation='nearest')
    plt.subplot(152)
    plt.imshow(img[1], interpolation='nearest')
    plt.subplot(153)
    plt.imshow(img[2], interpolation='nearest')
    plt.subplot(154)
    plt.imshow(img[3], interpolation='nearest')
    plt.subplot(155)
    plt.imshow(img[4], interpolation='nearest')
    plt.show()
"""
