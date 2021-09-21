import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPool2D, Flatten, Dropout, BatchNormalization, InputLayer
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import regularizers
from tensorflow.keras.optimizers import Adam

import cv2
import os
import numpy as np

labels = ['angry', 'disgusted', 'fearful',
          'happy', 'neutral', 'sad', 'surprised']
img_size = 48


def get_data(data_dir):
    data = []
    for label in labels:
        path = os.path.join(data_dir, label)
        class_num = labels.index(label)
        for img in os.listdir(path):
            try:
                img = cv2.imread(os.path.join(path, img), cv2.COLOR_BGR2GRAY)
                img_arr = img[..., ::-1]
                resized_arr = cv2.resize(img_arr, (img_size, img_size))
                data.append([resized_arr, class_num])
            except Exception as e:
                print(e)
    return np.array(data)


train = get_data('../archive/train')
val = get_data('../archive/validation')

# Data Process
x_train = []
y_train = []
x_val = []
y_val = []

for feature, label in train:
    x_train.append(feature)
    y_train.append(label)

for feature, label in val:
    x_val.append(feature)
    y_val.append(label)

# Normalize the data
x_train = np.array(x_train)
x_val = np.array(x_val)

x_train = x_train.reshape(x_train.shape[0], img_size, img_size, 1)
y_train = np.array(y_train)

x_val = x_val.reshape(x_val.shape[0], img_size, img_size, 1)
y_val = np.array(y_val)
X_train = x_train.astype('float32')
X_test = x_val.astype('float32')
x_train = (x_train) / 255
x_val = (x_val) / 255

# Data Augmentation
datagen = ImageDataGenerator(
    featurewise_center=False,  # set input mean to 0 over the dataset
    samplewise_center=False,  # set each sample mean to 0
    featurewise_std_normalization=False,  # divide inputs by std of the dataset
    samplewise_std_normalization=False,  # divide each input by its std
    zca_whitening=False,  # apply ZCA whitening
    # randomly rotate images in the range (degrees, 0 to 180)
    rotation_range=30,
    zoom_range=0.2,  # Randomly zoom image
    # randomly shift images horizontally (fraction of total width)
    width_shift_range=0.1,
    # randomly shift images vertically (fraction of total height)
    height_shift_range=0.1,
    horizontal_flip=True,  # randomly flip images
    vertical_flip=False)  # randomly flip images


datagen.fit(x_train)

# Model Creating
model = Sequential()

model.add(Conv2D(32, 3,
          activation="relu", padding="same", input_shape=(48, 48, 1)))
model.add(BatchNormalization())
model.add(MaxPool2D((2, 2), (2, 2)))

model.add(Conv2D(32, 3, activation="relu", padding="same"))
model.add(BatchNormalization())
model.add(MaxPool2D((2, 2), (2, 2)))

model.add(Conv2D(64, 3, activation="relu", padding="same",
          kernel_regularizer=regularizers.l2(0.01)))
model.add(BatchNormalization())
model.add(MaxPool2D((2, 2), (2, 2)))
model.add(Dropout(0.2))

model.add(Flatten())
model.add(Dense(128, activation="relu"))
model.add(Dropout(0.25))
model.add(Dense(7, activation="softmax"))

model.summary()
opt = Adam(lr=0.00001)
model.compile(optimizer=opt, loss=tf.keras.losses.SparseCategoricalCrossentropy(
    from_logits=True), metrics=["acc"])

history = model.fit(x_train, y_train, batch_size=32, epochs=500,
                    validation_data=(x_val, y_val))


acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs_range = range(500)

plt.figure(figsize=(15, 15))
plt.subplot(2, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(2, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.show()
model.save("my_model.h5")
