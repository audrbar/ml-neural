import tensorflow as tf
from tensorflow.keras.applications import VGG16
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.applications.vgg16 import preprocess_input, decode_predictions
from tensorflow.keras.models import Model
import numpy as np

# 1. Įkelkite modelį su iš anksto apmokytais svoriais
model = VGG16(weights='imagenet')

# 2. Įkelkite vaizdą,
image_path = '/Users/audrius/Documents/VCSPython/ml-neural/data/wolf.jpg'
image = load_img(image_path, target_size=(224, 224))
print(f"\nImage: {image}")
image_array = img_to_array(image)
print(f"\nImage To array: {image_array}")
image_array = preprocess_input(image_array)
print(f"\nImage Preprocessed Input: {image_array}")
image_array = tf.expand_dims(image_array, axis=0)
print(f"\nImage Expand Dims: {image_array}")

# 3. Prognozuokite vaizdo klasę
predictions = model.predict(image_array)
print(f"\nPredictions:\n{predictions}")

# 4. Dekoduokite prognozes
decoded_predictions = decode_predictions(predictions, top=3)
print("\nDecoded Predictions:")
for i, (imagenet_id, label, score) in enumerate(decoded_predictions[0]):
    print(f"{i + 1}. {label} ({score:.2f})")

model = Model(inputs=model.input, outputs=model.get_layer('fc2').output)
embedding_vector = model.predict(image_array)
print(f"Embedding vector: {embedding_vector}")
