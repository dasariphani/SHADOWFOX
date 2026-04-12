import tensorflow as tf
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.applications.efficientnet import preprocess_input, decode_predictions
import numpy as np

print("⏳ Downloading EfficientNet from internet... please wait!")

# This automatically downloads the model from internet
model = EfficientNetB0(
    weights='imagenet',  # Downloads pretrained weights from internet
    include_top=True     # Includes classification layer (1000 classes)
)

model.save('image_classifier.h5')
print("✅ EfficientNet downloaded and saved as image_classifier.h5")