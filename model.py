# Misc. Basic Libraries
import os
import random
import matplotlib.pyplot as plt

# Machine Learning Libraries
import tensorflow as tf
from tensorflow.keras import datasets, layers, models
from sklearn.metrics import roc_auc_score, roc_curve, auc

# Computer Vision, Data Storage, and Image Processing Libraries
import cv2 as cv
import numpy as np

# Hyperparameters
LEARNING_RATE = 0.0001
NUM_EPOCHS = 30
BATCH_SIZE = 1
input_shape = 256

# Construct model structure and return
def build_model():
    return models.Sequential([
        # Feature Extraction (Convolutional Layers)
        layers.Conv2D(16, (5, 5), activation='relu', input_shape=(input_shape, input_shape, 3)),
        layers.MaxPooling2D((5, 5)),
        layers.Conv2D(32, (5, 5), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(128, (3, 3), activation='relu'),
        layers.Conv2D(256, (3, 3), activation='relu'),

        # Output Layers
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dense(128, activation='relu'),
        layers.Dense(128, activation='relu'),
        layers.Dense(1, activation='sigmoid'),
    ])

# Load the images from the dataset
def load_images():
    images = []
    labels = []

    # For each class in the data folder
    directory = os.listdir('./ML Dataset')
    try:
        directory.remove('.DS_Store')
    except:
        print('noDSSTORE')
    for folder in directory:
        print('\t', folder)

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

    return images, labels

# Shuffle the data, and split into Train/Test
def shuffle_and_split(images, labels):
    temp = list(zip(images, labels))
    random.seed(12345)  # Ensure data is shuffled the same way every time
    random.shuffle(temp)
    images_shuffled, labels_shuffled = zip(*temp)

    # Slice data into train and testing sets
    train_images = images_shuffled[int(0.1 * len(images_shuffled)):]
    train_labels = labels_shuffled[int(0.1 * len(images_shuffled)):]
    test_images = images_shuffled[:int(0.1 * len(images_shuffled))]
    test_labels = labels_shuffled[:int(0.1 * len(images_shuffled))]
    train_images, train_labels, test_images, test_labels = np.asarray(train_images) / 255.0, \
                                                           np.asarray(train_labels), \
                                                           np.asarray(test_images) / 255.0, \
                                                           np.asarray(test_labels)

    return train_images, train_labels, test_images, test_labels

# Draw the ROC Curve for model and baseline
def draw_plots(model):
    y_pred = model.predict(train_images).ravel()
    fpr, tpr, _ = roc_curve(train_labels, y_pred)
    auc_train = auc(fpr, tpr)

    # Calculate False and True Positive Rates (Validation Set)
    y_pred_v = model.predict(test_images).ravel()
    fpr_v, tpr_v, _ = roc_curve(test_labels, y_pred_v)
    auc_v = auc(fpr_v, tpr_v)

    # Compare to Baseline Predictor
    baseline_pred = np.random.rand(len(y_pred)).astype(np.float32)  # Random predictions
    fpr_base, tpr_base, _ = roc_curve(train_labels, baseline_pred)  # ROC Curve Calculations
    auc_base = auc(fpr_base, tpr_base)  # AUC calculation

    # Plot Model AUC/ROC curves
    plt.title(f'ROC Curves (LR - {LEARNING_RATE})')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.plot(fpr, tpr, label='Training')
    plt.plot(fpr_v, tpr_v, label='Validation')
    plt.plot(fpr_base, tpr_base, label='Baseline Predictor')
    plt.plot((0, 1), (0, 1), '--', label='Baseline Approximation')
    plt.legend()
    plt.savefig(f'./graphs/{LEARNING_RATE}.png')
    plt.show()

rates = [0.0005]

for rate in rates:
    LEARNING_RATE = rate

    # Optimizer, Loss, and Gathered Metrics
    OPTIMIZER = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE)
    LOSS_MODEL = tf.losses.BinaryCrossentropy()
    METRICS = [
        'accuracy',
        tf.keras.metrics.MeanSquaredError(),
    ]
    class_names = ['Tulip', 'Lily'] # class_names = ['Carnation', 'Tulip', 'Daisy', 'Lily', 'Orchid', 'Rose']

    # Model Setup & Summary
    model = build_model()
    model.summary()

    # Load the data and shuffle it
    images, labels = load_images()

    train_images, train_labels, test_images, test_labels = shuffle_and_split(images, labels)

    # Draw a plot of randomly sampled data
    '''
    plt.figure(figsize=(10,10))
    for i in range(25):
        plt.subplot(5,5,i+1)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.imshow(train_images[i])
        plt.xlabel(class_names[train_labels[i][0]])
    plt.show()
    '''

    # Compile & Train Model
    model.compile(optimizer=OPTIMIZER, loss=LOSS_MODEL, metrics=METRICS)
    history = model.fit(train_images, train_labels, epochs=NUM_EPOCHS, validation_data=(test_images, test_labels))
    model.save(f'./models/model_a_{LEARNING_RATE}_e_{NUM_EPOCHS}_b_{BATCH_SIZE}.mod')
    model = tf.keras.models.load_model(f'./models/model_a_{LEARNING_RATE}_e_{NUM_EPOCHS}_b_{BATCH_SIZE}.mod')

    draw_plots(model)