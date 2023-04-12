# ECEN4493 HW4 Part 2
# Keenan Holsapple
# DNN for Wine library
# For libraries, "pip install -r requirements.txt"
# To execute, "python part2_keenan_holsapple.py"

# Outputs: procedural epoch information, loss/accuracy values, 
# loss+accuracy vs iterations graph

import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D
from tensorflow.keras.callbacks import EarlyStopping


# splitting incoming data
(X_train, y_train), (X_test, y_test) = mnist.load_data()

# Normalize pixel values to be between 0 and 1
X_train = X_train / 255.0
X_test = X_test / 255.0

# defining early stopping
early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

# defining dnn model at 7 layers
model = Sequential()

# input convolutional layer 1 (1)
model.add(Conv2D(64, (3,3), input_shape=(28, 28, 1)))
model.add(Activation("relu"))

# pooling layer 1 (2)
model.add(MaxPooling2D(pool_size=(2,2)))

# convolutional layer 2 (3)
model.add(Conv2D(64, (3,3)))
model.add(Activation("relu"))

# pooling layer 2 (4)
model.add(MaxPooling2D(pool_size=(2,2)))

# flattening layer 1 (5)
model.add(Flatten())

# dense layer 1 (6)
model.add(Dense(64, activation="relu"))

# dense layer 2 (7)
model.add(Dense(10, 
                # using softmax since 0-9 output measurements
                activation='softmax'))

# loss measured with multiple different output variables, adam optimizer, and measuring for accuracy in model
model.compile(loss="sparse_categorical_crossentropy", optimizer="adam", metrics=['accuracy'])

# training specifications
train = model.fit(X_train.reshape(-1, 28, 28, 1), y_train, 
                  
                  # reshaping: size input data, height, width, channel
                  validation_data=(X_test.reshape(-1, 28, 28, 1),y_test), 

                  # number of epochs
                  epochs=3, 

                  # stops overlearning in case
                  callbacks=[early_stop])

# finding loss and accuracy depending on inputs
loss, accuracy = model.evaluate(X_test.reshape(-1, 28, 28, 1), y_test)

# plot specifications
fig, ax = plt.subplots(figsize=(12,10))
plt.plot(train.history['accuracy'], label='accuracy')
plt.plot(train.history['val_accuracy'], label='val_accuracy', linestyle='--')
plt.plot(train.history['loss'], label='loss')
plt.plot(train.history['val_loss'], label='val_loss', linestyle='--')
plt.legend()
plt.xlabel("Number of Iterations")
plt.ylabel("Accuracy and Loss")
plt.title("Loss Regression Model for MNIST CNN")

# output to line value of loss and accuracy
print("Loss Value: " + str(loss))
print("Accuracy Value: " + str(accuracy))

# show plot
plt.show()