import numpy as np
from PIL import Image
from tensorflow import keras


class ImageRecognizer:
    def __init__(self, labels: list, model_path: str = "model.h5"):
        self.model = keras.models.load_model(model_path)
        self.input_size = self.model.layers[0].input_shape[1:3]
        self.labels = labels

    def predict(self, path: str) -> str:
        image = self._load_image(path)
        result = np.argmax(self.model.predict(image))
        return self.labels[result]

    def _load_image(self, path: str) -> np.array:
        image = Image.open(path)
        image = image.resize(self.input_size)
        image = np.array(image) / 255
        image = np.expand_dims(image, axis=0)

        return image
