from PIL import Image
import numpy as np
import tensorflow as tf
import numpy as np
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from tensorflow.keras.applications.resnet50 import preprocess_input, decode_predictions

# load image and convert to grayscale
img = Image.open('test_images/number9.jpg').convert('L')

model_path = 'mnist_cnn.h5'
model = load_model(model_path)

# resize image to 28x28 pixels
img = img.resize((28, 28))

# convert image to numpy array
img_arr = np.array(img)

# normalize pixel values to be between 0 and 1
img_arr = img_arr / 255.0

# reshape array to be a 4D tensor
img_arr = img_arr.reshape((1, 28, 28, 1))

# make prediction on image
prediction = model.predict(img_arr)

# print predicted label for image
print(np.argmax(prediction))