from model import classification_model  as c
import numpy as np
from PIL import Image
from io import BytesIO
import os


def load_model(weights_path, image_size):
    model = c.CNN(image_size)
    model.load_weights(weights_path)
    print("Model loaded")
    return model


image_size = 400
dir_path = os.path.dirname(os.path.realpath(__file__))
filepath = os.path.join(dir_path, 'final_model_w.h5')
print(dir_path)
print(filepath)
model = load_model(filepath, image_size)


def read_image(image_encoded):
    return Image.open(BytesIO(image_encoded))


def preprocess(image: Image.Image, image_size):
    image = image.resize((image_size, image_size))
    image = np.asfarray(image)
    image = image / 255.0
    image = np.expand_dims(image, 0)

    return image


def predict(image: np.ndarray):
    predications = model.predict(image)
    predicted = np.argmax(predications)
    class_dict = {0: 'Bed', 1: 'Chair', 2: 'Sofa'}
    for k, v in class_dict.items():
        if predicted == k:
            return v
