import tensorflow as tf
from tensorflow.keras.preprocessing import image
from model import build_model
import numpy as np
import json

BATCH_SIZE = 32
IMAGE_SIZE = 299
CHANNELS = 3
NUM_CLASSES = 12
INPUT_SHAPE = (BATCH_SIZE, IMAGE_SIZE, IMAGE_SIZE, CHANNELS)
class_names = ['Abnormal', 'Erythrodermic', 'Guttate', 'Inverse', 'Nail', 'Normal', 'Not Define', 'Palm Soles', 'Plaque', 'Psoriatic Arthritis', 'Pustular', 'Scalp']

def predict(model, img_path):
    img = image.load_img(img_path, target_size=(IMAGE_SIZE, IMAGE_SIZE))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0

    predictions = model.predict(img_array)
    predicted_class = class_names[np.argmax(predictions[0])]
    confidence = round(100 * (np.max(predictions[0])), 2)
    return predicted_class, confidence

if __name__ == "__main__":
    # Load model architecture from JSON
    with open("../models/Acc97.json", 'r') as json_file:
        model_json = json_file.read()
    model = tf.keras.models.model_from_json(model_json)

    # Load model weights
    model.load_weights("../models/Acc97.weights.h5")

    img_path = "../data/test_image.jpg"
    predicted_class, confidence = predict(model, img_path)
    print(f"Predicted class: {predicted_class} with confidence {confidence}%")
