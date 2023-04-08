# import tensorflow as tf
# from keras.datasets import mnist
# import matplotlib.pyplot as plt
# from sklearn.model_selection import train_test_split

# vine = tf.keras.datasets.mnist.load_data(path="mnist.npz")

# # X2_train = vine.data

# # X2_test = vine.target.astype('int')

# # X_train, X_test, y_train, y_test = train_test_split(X2_train, X2_test, test_size=0.3, random_state=0)


# (X_train, y_train), (X_test, y_test) = tf.keras.datasets.mnist.load_data()

# # (train_images, train_labels), (test_images, test_labels) = datasets.cifar10.load_data()

# # Normalize pixel values to be between 0 and 1
# X_train, y_train = X_train / 255.0, y_train / 255.0

# class_names = [0,1,2,3,4,5,6,7,8,9]

# plt.figure(figsize=(10,10))
# for i in range(25):
#     plt.subplot(5,5,i+1)
#     plt.xticks([])
#     plt.yticks([])
#     plt.grid(False)
#     plt.imshow(X_train[i])
#     # The CIFAR labels happen to be arrays, 
#     # which is why you need the extra index
#     plt.xlabel(class_names[X_test[i][0]])
# plt.show()


import tensorflow as tf
from keras.datasets import mnist
import matplotlib.pyplot as plt

(X_train, y_train), (X_test, y_test) = tf.keras.datasets.mnist.load_data()

# Normalize pixel values to be between 0 and 1
X_train = X_train / 255.0

plt.figure(figsize=(10,10))
for i in range(25):
    plt.subplot(5,5,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(X_train[i])
    plt.xlabel(y_train[i])
plt.show()

model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

history = model.fit(train_images, train_labels, epochs=10, 
                    validation_data=(test_images, test_labels))

plt.plot(history.history['accuracy'], label='accuracy')
plt.plot(history.history['val_accuracy'], label = 'val_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.ylim([0.5, 1])
plt.legend(loc='lower right')

test_loss, test_acc = model.evaluate(test_images,  test_labels, verbose=2)