import os

import numpy as np
import tensorflow as tf

from tensorflow.keras import datasets, layers, models
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt
import cv2 as cv

import random

def auroc(y_true, y_pred):
    return tf.py_function(roc_auc_score, (y_true, y_pred), tf.double)

images = []
labels = []

input_shape = 256

# Model Setup
model = models.Sequential()

# Convolutional Layers
# Equation - (((input dim - conv size) / stride) + 1)^2 x num_kernels
    # num_kernels == first parameter
model.add(layers.Conv2D(16, (5, 5), activation='relu', input_shape=(input_shape, input_shape, 3)))
model.add(layers.MaxPooling2D((5, 5)))
model.add(layers.Conv2D(32, (5, 5), activation='relu', input_shape=(input_shape, input_shape, 3)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(128, (3, 3), activation='relu'))
model.add(layers.Conv2D(256, (3, 3), activation='relu'))

# Output Layers
model.add(layers.Flatten())
model.add(layers.Dense(128, activation='relu'))
model.add(layers.Dense(128, activation='relu'))
model.add(layers.Dense(128, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))

# Summary Data
model.summary()

# For each class in the data folder
directory  = os.listdir('./ML Dataset')
try:
    directory.remove('.DS_Store')
except:
    print('noDSSTORE')
for folder in directory:
    print('\t',folder)

    # For each image in the class folder
    for image in os.listdir(f'./ML Dataset/{folder}'):
        if image != '.DS_Store':
            # Add image data to dataset
            try:

                # Add the image label to the list of labels
                # if folder == 'carnation':
                #     images.append(image_data_small)
                #     labels.append([0])
                # if folder == 'Rose-Flower':
                #     images.append(image_data_small)
                #     labels.append([5])
                if folder == 'tulips':
                    image_data = cv.imread(f'./ML Dataset/{folder}/{image}')

                    image_data_small = cv.resize(image_data, (input_shape, input_shape))
                    image_data_small = cv.cvtColor(image_data_small, cv.COLOR_BGR2RGB)
                    images.append(image_data_small)
                    labels.append([0])
                # if folder == 'daisy-flower':
                #     images.append(image_data_small)
                #     labels.append([2])
                if folder == 'Lily-flower':
                    image_data = cv.imread(f'./ML Dataset/{folder}/{image}')
                    image_data_small = cv.resize(image_data, (input_shape, input_shape))
                    image_data_small = cv.cvtColor(image_data_small, cv.COLOR_BGR2RGB)
                    images.append(image_data_small)
                    labels.append([1])
                # if folder == 'orchid':
                #     images.append(image_data_small)
                #     labels.append([4])

            except:
                print('I fucked up.')



# Shuffle the datasets
temp = list(zip(images, labels))
random.shuffle(temp)
images_shuffled, labels_shuffled = zip(*temp)

# Slice data into train and testing sets
train_images = images_shuffled[int(0.8 * len(images_shuffled)):]
train_labels = labels_shuffled[int(0.8 * len(images_shuffled)):]
test_images = images_shuffled[:int(0.8 * len(images_shuffled))]
test_labels = labels_shuffled[:int(0.8 * len(images_shuffled))]
train_images, train_labels, test_images, test_labels = np.asarray(train_images) / 255.0, \
                                                       np.asarray(train_labels), \
                                                       np.asarray(test_images) / 255.0, \
                                                       np.asarray(test_labels)

class_names = ['Tulip', 'Lily'] # class_names = ['Carnation', 'Tulip', 'Daisy', 'Lily', 'Orchid', 'Rose']

# Draw a plot of randomly sampled data
plt.figure(figsize=(10,10))
for i in range(25):
    plt.subplot(5,5,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(train_images[i])
    plt.xlabel(class_names[train_labels[i][0]])

# plt.show()

# Compile Model
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
              loss=tf.losses.BinaryCrossentropy(),
              metrics=['accuracy', auroc])

# Train Model
history = model.fit(train_images, train_labels, epochs=5, validation_data=(test_images, test_labels))

# Work with model history
print(history.params)
print(history.history.keys())
# print(history.history[])

pred_x = model.predict(test_images).ravel()

from sklearn.metrics import roc_curve
test_image_predictions = model.predict(test_images).ravel()
fpr_keras, tpr_keras, thresholds_keras = roc_curve(test_labels, test_image_predictions)

from sklearn.metrics import auc
auc_keras = auc(fpr_keras, tpr_keras)