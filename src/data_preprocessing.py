import tensorflow as tf
from tensorflow.keras import layers

BATCH_SIZE = 32
IMAGE_SIZE = 299
CHANNELS = 3

def load_dataset(data_dir, batch_size=BATCH_SIZE, img_size=IMAGE_SIZE):
    return tf.keras.preprocessing.image_dataset_from_directory(
        data_dir,
        seed=72,
        shuffle=True,
        image_size=(img_size, img_size),
        batch_size=batch_size
    )

def get_dataset_partitions_tf(ds, train_split=0.8, val_split=0.1, test_split=0.1, shuffle=True, shuffle_size=10000):
    assert (train_split + test_split + val_split) == 1
    ds_size = len(ds)
    if shuffle:
        ds = ds.shuffle(shuffle_size, seed=12)
    train_size = int(train_split * ds_size)
    val_size = int(val_split * ds_size)
    train_ds = ds.take(train_size)
    val_ds = ds.skip(train_size).take(val_size)
    test_ds = ds.skip(train_size).skip(val_size)
    return train_ds, val_ds, test_ds

def preprocess_data(ds):
    resize_and_rescale = tf.keras.Sequential([
        layers.Resizing(IMAGE_SIZE, IMAGE_SIZE),
        layers.Rescaling(1./255),
    ])
    data_augmentation = tf.keras.Sequential([
        layers.RandomFlip("horizontal_and_vertical"),
        layers.RandomRotation(0.2),
    ])
    return ds.map(lambda x, y: (resize_and_rescale(x), y)).map(lambda x, y: (data_augmentation(x, training=True), y))

def prepare_datasets(data_dir):
    dataset = load_dataset(data_dir)
    train_ds, val_ds, test_ds = get_dataset_partitions_tf(dataset)
    train_ds = preprocess_data(train_ds).cache().shuffle(1000).prefetch(buffer_size=tf.data.AUTOTUNE)
    val_ds = preprocess_data(val_ds).cache().shuffle(1000).prefetch(buffer_size=tf.data.AUTOTUNE)
    test_ds = preprocess_data(test_ds).cache().shuffle(1000).prefetch(buffer_size=tf.data.AUTOTUNE)
    return train_ds, val_ds, test_ds
