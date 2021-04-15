from typing import Tuple

from tensorflow import keras


def build_model(layers: list) -> keras.Model:
    model = keras.models.Sequential()

    for layer in layers:
        model.add(layer)

    model.compile(optimizer='adam', loss=keras.losses.SparseCategoricalCrossentropy(
        from_logits=True), metrics=['accuracy'])

    return model


def build_model_transfer(input_size: Tuple[int, int], output_layers: list) -> keras.Model:
    model_base = keras.applications.MobileNetV2(input_shape=(
        *input_size, 3), include_top=False, weights='imagenet')
    model_base.trainable = False

    model = keras.models.Sequential([model_base])
    for layer in output_layers:
        model.add(layer)

    model.compile(optimizer='adam', loss=keras.losses.SparseCategoricalCrossentropy(
        from_logits=True), metrics=['accuracy'])

    return model


def train_model(model: keras.Model, image_directory: str, classes: list, epochs: int = 16) -> None:
    train_image_dir = image_directory + "/train"
    test_image_dir = image_directory + "/test"

    image_size = model.layers[0].input_shape[1:3]

    batch_size_train = 32
    batch_size_test = 32

    generator = keras.preprocessing.image.ImageDataGenerator(
        rescale=1. / 255)

    generator_modified = keras.preprocessing.image.ImageDataGenerator(rescale=1. / 255.,
                                                                      rotation_range=45,
                                                                      width_shift_range=.15,
                                                                      height_shift_range=.15,
                                                                      horizontal_flip=True,
                                                                      zoom_range=0.5
                                                                      )

    train_data_gen = generator_modified.flow_from_directory(directory=train_image_dir,
                                                            batch_size=batch_size_train,
                                                            shuffle=True,
                                                            target_size=image_size,
                                                            classes=classes,
                                                            class_mode="sparse")
    val_data_gen = generator.flow_from_directory(directory=test_image_dir,
                                                 batch_size=batch_size_test,
                                                 shuffle=True,
                                                 target_size=image_size,
                                                 classes=classes,
                                                 class_mode="sparse")

    model.fit(train_data_gen, epochs=epochs,
              validation_data=val_data_gen)
