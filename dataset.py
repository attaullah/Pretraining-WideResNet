import tensorflow as tf
import tensorflow_datasets as tfds

# path to ImageNet dataset directory containing train/ and val/ directories.
base_dir = '../../ml/datasets/ImageNet/'
# automatically determines the number of parallel calls using tf.data API
AUTOTUNE = tf.data.AUTOTUNE


def normalize_img(image, label):
    """Normalizes images: `uint8` -> `float32`."""
    return tf.cast(image, tf.float32) / 255., label


def preprocess_image(image, label):
    i = image
    i = tf.cast(i, tf.float32) / 255.
    i = tf.image.resize_with_crop_or_pad(i, 224, 224)
    return i, label


def get_dataset(d, batch_size=128):
    """
    Prepares datasets ImageNet 224, 64, and 32.
    :param d: name of dataset one of: "imagenet_resized/32x32", "imagenet_resized/64x64", "imagenet-full"
    :param batch_size: size of mini-batch
    :return: train and test datasets, size of an image
    """
    if 'imagenet-full' in d:
        size = 224
        builder = tfds.ImageFolder(base_dir)
        ds_train, ds_test = builder.as_dataset(split=['train', 'val'], shuffle_files=True, as_supervised=True)

    else:
        ds_train, ds_test = tfds.load(d, split=['train', 'validation'], shuffle_files=True, as_supervised=True)

        if "64" in d:
            size = 64
        else:
            size = 32

    if 'imagenet-full' in d:
        ds_train = ds_train.map(preprocess_image, num_parallel_calls=AUTOTUNE)
        ds_test = ds_test.map(preprocess_image, num_parallel_calls=AUTOTUNE)
    else:
        ds_train = ds_train.map(normalize_img, num_parallel_calls=AUTOTUNE)
        ds_test = ds_test.map(normalize_img, num_parallel_calls=AUTOTUNE)

    ds_train = ds_train.batch(batch_size)
    ds_test = ds_test.batch(batch_size)

    ds_train = ds_train.prefetch(AUTOTUNE)
    ds_test = ds_test.prefetch(AUTOTUNE)
    return ds_train, ds_test, size
