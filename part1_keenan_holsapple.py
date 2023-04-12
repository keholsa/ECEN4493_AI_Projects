# ECEN4493 HW4 Part 1
# Keenan Holsapple
# DNN for Wine library
# For libraries, "pip install -r requirements.txt"
# To execute, "python part1_keenan_holsapple.py"

# Outputs: procedural epoch information, loss/accuracy values, 
# loss+accuracy vs iterations graph

import matplotlib.pyplot as plt
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from keras.models import Sequential
from keras.layers import Dense
from tensorflow.keras.callbacks import EarlyStopping

# loading in dataset from website
vine = load_wine()

# defining x2_train to hold all data
X2_train = vine.data

# defining x2_test to hold all test values
X2_test = vine.target.astype('int')

# splitting data into training values
X_train, X_test, y_train, y_test = train_test_split(X2_train, X2_test, test_size=0.3, random_state=0)

# scaling data using existing library; keeps learning consistent
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# stops model from overlearning
early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

# defining neural network with layers and neurons
model = Sequential([
# input layer
Dense(units=(X_train.shape[1]), input_shape=(X_train.shape[1],), activation='relu'),
# layer 1
Dense(units=64, activation='relu'),
# layer 2
Dense(units=128, activation='relu'),
# layer 3
Dense(units=256, activation='relu')
])


# compiler defining adam as the optimiser
# loss needs to be sparse_categorical_crossentropy since out since test values aren't binary, measures accuracy
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# training the model with data at 30 epochs fixed, incoming test data form wine list, and early stop defined
train = model.fit(X_train, y_train, validation_data=(X_test,y_test), epochs=30, callbacks=[early_stop])

# computing values for loss and accuracy in the model
loss, accuracy = model.evaluate(X_test, y_test)

# defining plot figure
fig, ax = plt.subplots(figsize=(10,10))

# labeling plot
plt.plot(train.history['accuracy'], label='accuracy')
# measuring accuracy of accuracy
plt.plot(train.history['val_accuracy'], label='val_accuracy', linestyle='--')
plt.plot(train.history['loss'], label='loss')
# measuing accuracy of loss
plt.plot(train.history['val_loss'], label='val_loss', linestyle='--')
plt.legend()
plt.xlabel("Number of Iterations")
plt.ylabel("Accuracy and Loss")
plt.title("Loss Regression Model for Wine DNN")


# output to line value of loss and accuracy
print("Loss Value: " + str(loss))
print("Accuracy Value: " + str(accuracy))

# show plot
plt.show()
