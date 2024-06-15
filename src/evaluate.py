import tensorflow as tf
from data_preprocessing import prepare_datasets
from model import build_model
from sklearn.metrics import confusion_matrix, classification_report
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import json

BATCH_SIZE = 32
IMAGE_SIZE = 299
CHANNELS = 3
NUM_CLASSES = 12
INPUT_SHAPE = (BATCH_SIZE, IMAGE_SIZE, IMAGE_SIZE, CHANNELS)
class_names = ['Abnormal', 'Erythrodermic', 'Guttate', 'Inverse', 'Nail', 'Normal', 'Not Define', 'Palm Soles', 'Plaque', 'Psoriatic Arthritis', 'Pustular', 'Scalp']

def evaluate_model(data_dir, model_architecture, model_weights):
    _, _, test_ds = prepare_datasets(data_dir)

    # Load model architecture from JSON
    with open(model_architecture, 'r') as json_file:
        model_json = json_file.read()
    model = tf.keras.models.model_from_json(model_json)

    # Load model weights
    model.load_weights(model_weights)

    scores = model.evaluate(test_ds)
    print(f"Test Accuracy: {scores[1] * 100:.2f}%")

    y_pred = []
    y_true = []
    for images, labels in test_ds:
        y_pred.extend(np.argmax(model.predict(images), axis=1))
        y_true.extend(labels.numpy())

    cm = confusion_matrix(y_true, y_pred)
    np.save('../models/cm.npy', cm)

    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, cmap="Blues", xticklabels=class_names, yticklabels=class_names)
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted Label")
    plt.ylabel("Actual Label")
    plt.show()

    print(classification_report(y_true, y_pred, target_names=class_names))

if __name__ == "__main__":
    data_dir = "../data/Full Data Class"
    model_architecture = "../models/Acc97.json"
    model_weights = "../models/Acc97.weights.h5"
    evaluate_model(data_dir, model_architecture, model_weights)
