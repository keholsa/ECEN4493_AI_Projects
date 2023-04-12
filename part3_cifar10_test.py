import tensorflow as tf
import numpy as np
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from tensorflow.keras.applications.resnet50 import preprocess_input, decode_predictions

# Load the CIFAR-10 CNN model
model_path = 'cifar10_cnn.h5'
model = load_model(model_path)

# Load and preprocess the test image
img_path = 'bird_1.jpg'
img = load_img(img_path, target_size=(32, 32))
x = img_to_array(img)
x = np.expand_dims(x, axis=0)
x = preprocess_input(x)

# Make a prediction
preds = model.predict(x)
print('Predicted:', preds)

# Decode the prediction result
class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck']
pred_class = np.argmax(preds)
pred_label = class_names[pred_class]
print('Predicted label:', pred_label)