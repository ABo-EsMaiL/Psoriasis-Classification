import tensorflow as tf
from data_preprocessing import prepare_datasets
from model import build_model

BATCH_SIZE = 32
IMAGE_SIZE = 299
CHANNELS = 3
EPOCHS = 100
NUM_CLASSES = 12
INPUT_SHAPE = (BATCH_SIZE, IMAGE_SIZE, IMAGE_SIZE, CHANNELS)

def train_model(data_dir):
    train_ds, val_ds, _ = prepare_datasets(data_dir)
    model = build_model(INPUT_SHAPE, NUM_CLASSES)
    model.compile(
        optimizer='adam',
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
        metrics=['accuracy']
    )
    history = model.fit(
        train_ds,
        batch_size=BATCH_SIZE,
        validation_data=val_ds,
        epochs=EPOCHS,
    )

    # Save model architecture to JSON
    model_json = model.to_json()
    with open('../models/Acc97.json', 'w') as json_file:
        json_file.write(model_json)

    # Save model weights
    model.save_weights('../models/Acc97.weights.h5')

    return history, model

if __name__ == "__main__":
    data_dir = "../data/Full Data Class"
    history, model = train_model(data_dir)
