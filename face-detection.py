import os
from keras.preprocessing import image

!pip install opencv-python

import cv2

categories = ["without_mask", "with_mask"]
data = []

for category in categories:
    path = os.path.join('Dataset', category)
    label = categories.index(category)
    for file in os.listdir(path):
        img_path = os.path.join(path, file)
        img = cv2.imread(img_path)
        img = cv2.resize(img, (224, 224))
        data.append([img, label])

len(data)

import random
random.shuffle(data)

X = []
y = []

for features, label in data:
    X.append(features)
    y.append(label)

len(X)
len(y)

import numpy as np
X = np.array(X)
y = np.array(y)

X.shape
y.shape

X = X / 255

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

from keras.applications import VGG16
vgg = VGG16()
vgg.summary()

from keras import Sequential
model = Sequential()
for layer in vgg.layers[:-1]:
    model.add(layer)

model.summary()

for layer in model.layers:
    layer.trainable = False

model.summary()

from keras.layers import Dense
model.add(Dense(1, activation='sigmoid'))
model.summary()

model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

model.fit(X_train, y_train, epochs=1, validation_data=(X_test, y_test))

model.save('model.h5')
