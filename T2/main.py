import tensorflow as tf
import numpy as np
from tensorflow.keras.utils import load_img, img_to_array
import json

with open('class_labels.json', 'r') as f:
    class_labels = json.load(f)

model = tf.keras.models.load_model('model.keras')

image_path = '/content/valid/1-00/41.jpg' # Replace with the path to your test image
img = load_img(image_path, target_size=(224, 224))  # Load the image and resize it to match the input size of your model
img_array = img_to_array(img)  # Convert the image to a NumPy array
img_array = np.expand_dims(img_array, axis=0)  # Add an extra dimension to represent the batch size (1 in this case)
img_array = img_array / 255.0  # Normalize the image data

prediction = model.predict(img_array)
predicted_class_index = np.argmax(prediction)
predicted_class_label = class_labels[predicted_class_index]

print(f"Predicted class: {predicted_class_label}")