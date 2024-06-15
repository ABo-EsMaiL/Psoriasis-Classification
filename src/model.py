from tensorflow.keras import layers, models
from keras.applications import Xception

def build_model(input_shape, num_classes):
    xception = Xception(
        include_top=False,
        weights='imagenet',
        input_shape=(input_shape[1], input_shape[2], input_shape[3])
    )

    for layer in xception.layers[:-25]:
        layer.trainable = False

    model = models.Sequential([
        xception,
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dense(num_classes, activation='softmax'),
    ])

    model.build(input_shape=input_shape)
    return model
