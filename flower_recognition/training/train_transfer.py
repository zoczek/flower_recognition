from tensorflow import keras

from model.model_training import build_model_transfer, train_model

input_size = (64, 64)

classes = ["Daffodil", "Snowdrop", "LilyValley", "Bluebell", "Crocus", "Iris", "Tigerlily", "Tulip",
           "Fritillary", "Sunflower", "Daisy", "ColtsFoot", "Dandelion", "Cowslip", "Buttercup", "Windflower",
           "Pansy"]

output_layers = [
    keras.layers.GlobalAveragePooling2D(),
    keras.layers.Dense(len(classes))
]

model = build_model_transfer(input_size, output_layers)
train_model(model, "flowers", classes)
model.save("model.h5")
