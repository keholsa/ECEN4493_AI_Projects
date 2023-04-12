import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D

import pickle
from pathlib import Path

HERE = Path(__file__).parent

X = pickle.load(open(HERE / "X.pickle","rb"))
y = pickle.load(open(HERE / "y.pickle","rb"))

# finding max and min of pixel data, keras has normalize library?
X = X / 255.0


model = Sequential()

# convlutional layer, window size, input shape
model.add(Conv2D(64, (3,3), input_shape = X.shape[1:]))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(64, (3,3)))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(2,2)))

# converting 3d to 1d
model.add(Flatten())

model.add(Dense(64))

model.add(Dense(1))
model.add(Activation('sigmoid'))

model.compile(loss="binary_crossentropy", optimizer="adam", metrics=['accuracy'])

# number of samples passed at a time
model.fit(X, y, batch_size=32, validation_split=0.1, epochs=10)



