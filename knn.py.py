import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import f1_score, confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
import cv2 as cv

input_shape = 256
images = []
labels = []

# For each class in the data folder
directory = os.listdir('./sample-images')
try:
    directory.remove('.DS_Store')
except:
    print('noDSSTORE')
for folder in directory:
    print('\t', folder)

    # For each image in the class folder
    for image in os.listdir(f'./sample-images/{folder}'):
        if image != '.DS_Store':
            # Add image data to dataset
            try:

                # Add the image label to the list of y
                # if folder == 'carnation':
                #     X.append(image_data_small)
                #     y.append([0])
                # if folder == 'Rose-Flower':
                #     X.append(image_data_small)
                #     y.append([5])
                if folder == 'tulips':
                    image_data = cv.imread(f'./sample-images/{folder}/{image}')
                    image_data_small = cv.resize(image_data, (input_shape, input_shape))
                    image_data_small = cv.cvtColor(image_data_small, cv.COLOR_BGR2GRAY)
                    images.append(image_data_small)
                    labels.append([0])
                # if folder == 'daisy-flower':
                #     X.append(image_data_small)
                #     y.append([2])
                if folder == 'lily-flowers':
                    image_data = cv.imread(f'./sample-images/{folder}/{image}')
                    image_data_small = cv.resize(image_data, (input_shape, input_shape))
                    image_data_small = cv.cvtColor(image_data_small, cv.COLOR_BGR2GRAY)
                    images.append(image_data_small)
                    labels.append([1])
                # if folder == 'orchid':
                #     X.append(image_data_small)
                #     y.append([4])
            except:
                print('Didint work')
X = np.array(images)
y = np.array(labels)


def knn_research(X, y):
    n_range = [1, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50]

    X = np.reshape(X, (X.shape[0], -1))
    y = y.ravel()

    mean_error = []
    std_error = []
    for n in n_range:
        model = KNeighborsClassifier(n_neighbors=n, weights="uniform")
        temp = []
        kf = KFold(n_splits=5)
        for train, test in kf.split(X):
            model.fit(X[train], y[train])
            y_pred = model.predict(X[test])
            temp.append(f1_score(y[test], y_pred))
        print(max(temp))
        print(confusion_matrix(y[test], y_pred))
        mean_error.append(np.array(temp).mean())
        std_error.append(np.array(temp).std())
    plt.errorbar(n_range, mean_error, yerr=std_error)
    plt.xlabel("n")
    plt.ylabel("F1 Score")
    plt.xlim(1, 50)
    plt.show()


def knn():
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    # Reshape the image data into rows
    X_train = np.reshape(X_train, (X_train.shape[0], -1))
    X_test = np.reshape(X_test, (X_test.shape[0], -1))

    print("X_train: " + str(X_train.shape))
    print("X_test: " + str(X_test.shape))
    print("y_train: " + str(y_train.shape))
    print("y_test: " + str(y_test.shape))

    model = KNeighborsClassifier(n_neighbors=15, weights="uniform")
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    print(tn, fp, fn, tp)
    fig = plt.figure(figsize=(1, 1))
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(X_train[:, 0], X_train[:, 1], y_train, c="Blue")
    ax.scatter(X_test[:, 0], X_test[:, 1], y_pred, c="Green")
    ax.set_xlabel("x_1")
    ax.set_ylabel("x_2")
    ax.set_zlabel("y")
    plt.show()


if __name__ == '__main__':
    knn()
    knn_research(X, y)
