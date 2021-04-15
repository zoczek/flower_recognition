from tensorflow import keras

from model.model_training import build_model, train_model

input_size = (64, 64)

classes = ["Daffodil", "Snowdrop", "LilyValley", "Bluebell", "Crocus", "Iris", "Tigerlily", "Tulip",
           "Fritillary", "Sunflower", "Daisy", "ColtsFoot", "Dandelion", "Cowslip", "Buttercup", "Windflower",
           "Pansy"]

layers = [
    keras.layers.Conv2D(32, (3, 3), activation='relu',
                        input_shape=(*input_size, 3)),
    keras.layers.MaxPooling2D((2, 2)),
    keras.layers.Conv2D(32, (3, 3), activation='relu'),
    keras.layers.MaxPooling2D((2, 2)),
    keras.layers.Conv2D(32, (3, 3), activation='relu'),
    keras.layers.Dropout(0.4),

    keras.layers.Flatten(),
    keras.layers.Dense(64, activation='relu'),
    keras.layers.Dense(len(classes)),
]

model = build_model(layers)
train_model(model, "flowers", classes)

model.save("model.h5")
