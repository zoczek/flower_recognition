from model.recognizer import ImageRecognizer

labels = ["Daffodil", "Snowdrop", "LilyValley", "Bluebell", "Crocus", "Iris", "Tiger Lily", "Tulip",
          "Fritillary", "Sunflower", "Daisy", "Colt's Foot", "Dandelion", "Cowslip", "Buttercup",
          "Windflower", "Pansy"]

recognizer = ImageRecognizer(labels, "model.h5")
print(recognizer.predict('flower.jpg'))
