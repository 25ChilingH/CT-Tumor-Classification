# helper libraries
import numpy as np
import pandas as pd
import os

# plotting
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
import matplotlib.image as mpimg
import cv2
import seaborn as sns

# data preprocessing 
from sklearn.model_selection import train_test_split

# model building
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import MaxPooling2D, Dense, Dropout,Flatten, Conv2D
from keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ReduceLROnPlateau

# model evaluation
import time

tumor_dir = '../input/brian-tumor-dataset/Brain Tumor Data Set/Brain Tumor Data Set/Brain Tumor'
control_dir = '../input/brian-tumor-dataset/Brain Tumor Data Set/Brain Tumor Data Set/Healthy'
filepaths = []
targets = []
for root, _, filenames in os.walk(control_dir):
    for fName in filenames:
        file_path = os.path.join(root, fName)
        filepaths.append(file_path)
        targets.append("control")


for root, _, filenames in os.walk(tumor_dir):
    for fName in filenames:
        file_path = os.path.join(root, fName)
        filepaths.append(file_path)
        targets.append("tumor")

series = pd.Series(filepaths, name="filepaths")
series2 = pd.Series(targets, name="targets")
tumor_df = pd.concat([series, series2], axis=1)
df = pd.DataFrame(tumor_df)


_, testDf = train_test_split(df, test_size=0.3, random_state=42, shuffle= True)
trainDf, valDf = train_test_split(df, test_size=0.2, random_state=42, shuffle=True)

gen = ImageDataGenerator(
                            horizontal_flip=True,
                            preprocessing_function= tf.keras.applications.mobilenet_v2.preprocess_input
                        )
# scaled between [-1, 1]
train = gen.flow_from_dataframe(dataframe = trainDf, x_col="filepaths",y_col="targets",
                                      target_size=(244,244),
                                      color_mode='rgb',
                                      class_mode="categorical",
                                      batch_size=32,
                                      shuffle=False
                                     )

val = gen.flow_from_dataframe(dataframe= valDf,x_col="filepaths", y_col="targets",
                                    target_size=(244,244),
                                    color_mode= 'rgb',
                                    class_mode="categorical",
                                    batch_size=32,
                                    shuffle=False
                                   )

testGen = ImageDataGenerator(preprocessing_function= tf.keras.applications.mobilenet_v2.preprocess_input)
test = testGen.flow_from_dataframe(dataframe = testDf, x_col="filepaths", y_col="targets",
                                     target_size=(244,244),
                                     color_mode='rgb',
                                     class_mode="categorical",
                                     batch_size=32,
                                     shuffle= False
                                    )

model = Sequential()
model.add(Conv2D(filters=32,kernel_size=(3,3),strides=(1,1), activation="relu", padding="valid",
               input_shape=(244,244,3)))
model.add(MaxPooling2D(strides=2))
model.add(Conv2D(filters=48, kernel_size=(5,5), padding='valid', activation='relu'))
model.add(MaxPooling2D(strides=2))
model.add(Flatten())
model.add(Dropout(0.5))
model.add(Dense(256, activation='relu'))
model.add(Dense(84, activation='relu'))
model.add(Dense(2, activation='softmax'))
model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

model.summary()

start = time.time()
rlr = ReduceLROnPlateau(monitor='val_accuracy', verbose=1, mode='max',factor=0.3, min_lr=1e-6, patience=2)

history = model.fit(train, epochs=10, validation_data=val, verbose=1, callbacks=[rlr])
print("Total time: ", time.time() - start, "seconds")
model.evaluate(test)

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

plotHist(history)
y_pred = model.predict(test)
y_pred_max = np.argmax(y_pred, axis=1)

labels = (train.class_indices)
labels = dict((v,k) for k,v in labels.items())
y_pred = [labels[k] for k in y_pred_max]

y_test = testDf.targets
cf_matrix = confusion_matrix(y_true=y_test, y_pred=y_pred)
ax = sns.heatmap(cf_matrix, annot=True, cmap='Blues', fmt='g')
print("Accuracy of the Model:",accuracy_score(y_test, y_pred)*100,"%")
print(classification_report(y_test, y_pred))

ax.set_title('Brain Tumor CT Scans Confusion Matrix \n\n')
ax.set_xlabel('\nPredicted Tumor Present')
ax.set_ylabel('Actual Tumor Present ')

ax.xaxis.set_ticklabels(['False','True'])
ax.yaxis.set_ticklabels(['False','True'])

plt.show()