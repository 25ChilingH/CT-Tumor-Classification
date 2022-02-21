# helper libraries
import numpy as np
import pandas as pd
from time import strftime
import itertools
import random
import os
from os import walk
from os.path import join

# data augmentation
import albumentations as A

# plotting
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import matplotlib.image as mpimg
import cv2
import seaborn as sns

# data preprocessing 
from sklearn.model_selection import train_test_split

# model building
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Activation, BatchNormalization, Flatten, Dropout
from sklearn.utils import class_weight

# model evaluation
from sklearn.model_selection import KFold

METADATA_CSV = "../input/brian-tumor-dataset/metadata.csv"
TUMOR = '../input/brian-tumor-dataset/Brain Tumor Data Set/Brain Tumor Data Set/Brain Tumor/'
CONTROL = "../input/brian-tumor-dataset/Brain Tumor Data Set/Brain Tumor Data Set/Healthy/"

def augment_data(image):
    transform = A.Compose([
        A.RandomRotate90(),
        A.Flip(),
        A.Transpose(),
        A.OneOf([
            A.MotionBlur(p=.2),
            A.MedianBlur(blur_limit=3, p=0.3),
            A.Blur(blur_limit=3, p=0.1),
        ], p=0.2),
        A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.2, rotate_limit=45, p=0.2),
        A.OneOf([
            A.OpticalDistortion(p=0.3),
            A.GridDistortion(p=.1),
        ], p=0.2),
        A.RandomBrightnessContrast(),            
    ])
    transform2 = A.Compose([
        A.RandomRotate90(),
        A.Flip(),
        A.Transpose(),
        A.GaussNoise(),
        A.OneOf([
            A.MotionBlur(p=.2),
            A.MedianBlur(blur_limit=3, p=0.1),
            A.Blur(blur_limit=3, p=0.1),
        ], p=0.2),
        A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.2, rotate_limit=45, p=0.2),
        A.OneOf([
            A.OpticalDistortion(p=0.3),
            A.GridDistortion(p=.1),
            A.PiecewiseAffine(p=0.3),
        ], p=0.2),
        A.OneOf([
            A.Sharpen(),
            A.Emboss(),
            A.RandomBrightnessContrast(),            
        ], p=0.3),
    ])
    random.seed(42)
    return [transform(image=image)['image'], transform2(image=image)['image']]
    

# the dataset has both grayscale and RGB images so we need to convert RGB images to single channels
starting_data = pd.read_csv(METADATA_CSV)
def rgb2gray(img):
    output_img = img
    if img.ndim == 3:
        R, G, B = img[:, :, 0], img[:, :, 1], img[:, :, 2]
        output_img = 0.2989 * R + 0.5870 * G + 0.1140 * B # converting the channels depending on diff weights 
    if output_img.max() < 1.1:
        output_img = output_img * 255
    return output_img

def reduce_noise(noisy_img):
    clean_img = np.array(noisy_img)
    clean_img[noisy_img < 10] = 0 # reduces negligible parts to 0
    return clean_img

# some of the data have white white borders
def remove_white_borders(img_with_borders):
    # left
    if np.mean(img_with_borders[:, 0]) > 30:
        img_with_borders[:, 0] = 0;
    # right
    if np.mean(img_with_borders[:, -1]) > 30:
        img_with_borders[:, -1] = 0;
    # top
    if np.mean(img_with_borders[0, :]) > 30:
        img_with_borders[0, :] = 0;
    # bottom
    if np.mean(img_with_borders[-1, :]) > 30:
        img_with_borders[-1, :] = 0;
    return img_with_borders

# all part of data normalization
def remove_black_padding(img_with_padding):
    # Sides
    vert_mean = np.mean(img_with_padding, axis=0)
    vert_map = vert_mean > 1;
    vert_matches = np.where(vert_map == True)
    img_no_padding = img_with_padding[:, vert_matches[0][0]:vert_matches[0][-1]]
    # Top & bottom
    hori_mean = np.mean(img_no_padding, axis=1)
    hori_map = hori_mean > 1;
    hori_matches = np.where(hori_map == True)
    img_no_padding = img_no_padding[hori_matches[0][0]:hori_matches[0][-1], :]
    return img_no_padding

def square_image(rect_img):
    X, Y = rect_img.shape
    if X != Y:
        if X > Y:
            out_img = np.zeros((X, X))
            offset = int(np.ceil((X-Y) / 2))
            out_img[:, offset : (offset + Y)] = rect_img
        else:
            out_img = np.zeros((Y, Y))
            offset = int(np.ceil((Y-X) / 2))
            out_img[offset : (offset + X), :] = rect_img
        return out_img
    else:
        return rect_img

def preprocess_img(input_img):
    output_img = rgb2gray(input_img)
    output_img = reduce_noise(output_img)
    output_img = remove_white_borders(output_img)
    output_img = remove_black_padding(output_img)
    output_img = square_image(output_img)
    output_img = cv2.resize(output_img, dsize=(64, 64), interpolation=cv2.INTER_LINEAR)
    return output_img

features = np.zeros((1, 4096))
target = np.array([])
debug=1
# Fill with tumor images
for root, dirnames, filenames in walk(TUMOR):
    n_total = len(filenames)
    n = 1
    for filename in filenames:
        file_path = join(root, filename)
        n += 1       
        img = mpimg.imread(file_path)
        if n % 50 == 0:
            imgs = augment_data(img)
            for image in imgs:
                img = preprocess_img(image)
                features = np.vstack([features, np.reshape(img, -1)])
                target = np.append(target, 1)
            n+=2
            print(file_path)
        img = preprocess_img(img)
        features = np.vstack([features, np.reshape(img, -1)])
        target = np.append(target, 1)
features = features[1:, :]

for root, dirnames, filenames in walk(CONTROL):
    n_total = len(filenames)
    n = 1
    for filename in filenames:
        file_path = join(root, filename)
        n += 1       
        img = mpimg.imread(file_path)
        if n % 50 == 0:
            imgs = augment_data(img)
            for image in imgs:
                img = preprocess_img(image)
                features = np.vstack([features, np.reshape(img, -1)])
                target = np.append(target, 0)
            n+=2
            print(file_path)
        img = preprocess_img(img)
        features = np.vstack([features, np.reshape(img, -1)])
        target = np.append(target, 0)
        
df = pd.DataFrame(features)
df["target"] = target

target = df["target"]
features = df.drop("target", axis=1)
x_train, x_test, y_train, y_test = train_test_split(features, target, test_size=0.1, random_state=42, stratify=target)

model = Sequential()

model.add(BatchNormalization())
model.add(Dense(256, input_dim=4096, activation='relu'))
model.add(BatchNormalization())
model.add(Dense(128, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(BatchNormalization())
model.add(Dense(2, activation='softmax'))

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

history = model.fit(x_train, y_train, epochs=70, batch_size=100, validation_split=0.13, shuffle=True)
scores = model.evaluate(x_test, y_test)


def plotHist(histories):
    plt.plot(histories.history['accuracy'])
    plt.plot(histories.history['val_accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')
    plt.show()
    
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')
    plt.show()

model.summary()
plotHist(history)

y_pred = model.predict(x_test)
y_pred_max = np.argmax(y_pred, axis=1)
cf_matrix = confusion_matrix(y_true=y_test, y_pred=y_pred_max)
ax = sns.heatmap(cf_matrix, annot=True, cmap='Blues')

ax.set_title('Brain Tumor CT Scans Confusion Matrix \n\n');
ax.set_xlabel('\nPredicted Tumor Present')
ax.set_ylabel('Actual Tumor Present ');

ax.xaxis.set_ticklabels(['False','True'])
ax.yaxis.set_ticklabels(['False','True'])

plt.show()

