import os
import numpy as np
import tensorflow as tf
import cv2

from PIL import Image, ImageTk

from tensorflow.keras.models import model_from_json
from image_manipulation import modified_guided_grad_cam, guided_grad_cam


def load_model(clahe):
    json_file = open("models/efficientnetv2-S_model_architecture.json", "r")
    model_json = json_file.read()
    model = model_from_json(model_json)

    if clahe:
        model.load_weights("models/efficientnetv2-S_with_clahe_weights.h5")
    else:
        model.load_weights("models/efficientnetv2-S_without_clahe_weights.h5")

    return model


def classify(model, img):
    new_img = np.expand_dims(img, axis=0)
    probability = model.predict(new_img).ravel()

    classification = 1 if probability.mean() > 0.5 else 0

    print("prob:", probability)
    print("class:", classification)

    heatmap = guided_grad_cam(img, model)
    resized_img = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
    heatmap_img = Image.fromarray(resized_img)

    if classification == 1:
        label = "abnormal"
    else:
        label = "normal"

    return (probability[0], label, heatmap_img)
