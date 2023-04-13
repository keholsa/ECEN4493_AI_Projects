from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import preprocess_input, decode_predictions
import numpy as np


model = ResNet50(weights='imagenet')

# test image(can implement loop for reading multitude as well)
img_path = 'test_images/car.jpg'
# adapting image to test
img = image.load_img(img_path, target_size=(224, 224))
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)
x = preprocess_input(x)

# prediction
preds = model.predict(x)

# decoding prediction with embedded function in resnet50
print('Predicted:', decode_predictions(preds, top=3)[0])
