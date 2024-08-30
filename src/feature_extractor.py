import numpy as np
from PIL import Image
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.models import Model

from data_loader import read_images_from_dir, smart_crop


class FeatureExtractor:
    def __init__(self):
        self.model = self.get_model()

    def get_model(self):
        model = ResNet50(weights="imagenet", include_top=False)
        model = Model(inputs=model.input, outputs=model.layers[-1].output)
        return model

    def image_to_tensor(self, image_dir):
        images = read_images_from_dir(image_dir)
        images = [smart_crop(image) for image in images]
        images = [np.array(image) for image in images]
        images = [Image.fromarray(image).resize((224, 224)) for image in images]
        images = np.stack(images)
        return images

    def flatten_features(self, features):
        return features.reshape(features.shape[0], 7 * 7 * 2048)

    def extract_features(self, image_dir):
        images = self.image_to_tensor(image_dir)
        images = preprocess_input(images)
        features = self.model.predict(images)
        features = self.flatten_features(features)
        return features
