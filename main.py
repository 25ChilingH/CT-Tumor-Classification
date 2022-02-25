# %% [code] {"execution":{"iopub.status.busy":"2022-02-24T02:19:03.222374Z","iopub.execute_input":"2022-02-24T02:19:03.22278Z","iopub.status.idle":"2022-02-24T02:19:06.078921Z","shell.execute_reply.started":"2022-02-24T02:19:03.222679Z","shell.execute_reply":"2022-02-24T02:19:06.077817Z"}}
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

# %% [code] {"execution":{"iopub.status.busy":"2022-02-24T02:19:11.22876Z","iopub.execute_input":"2022-02-24T02:19:11.229065Z","iopub.status.idle":"2022-02-24T02:19:12.043279Z","shell.execute_reply.started":"2022-02-24T02:19:11.229034Z","shell.execute_reply":"2022-02-24T02:19:12.042254Z"}}
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

# %% [code] {"execution":{"iopub.status.busy":"2022-02-24T02:19:15.315225Z","iopub.execute_input":"2022-02-24T02:19:15.315608Z","iopub.status.idle":"2022-02-24T02:19:15.336333Z","shell.execute_reply.started":"2022-02-24T02:19:15.315559Z","shell.execute_reply":"2022-02-24T02:19:15.335566Z"}}
_, testDf = train_test_split(df, test_size=0.3, random_state=42, shuffle= True)
trainDf, valDf = train_test_split(df, test_size=0.2, random_state=42, shuffle=True)

# %% [code] {"execution":{"iopub.status.busy":"2022-02-24T02:19:25.187529Z","iopub.execute_input":"2022-02-24T02:19:25.187906Z","iopub.status.idle":"2022-02-24T02:19:26.00691Z","shell.execute_reply.started":"2022-02-24T02:19:25.187866Z","shell.execute_reply":"2022-02-24T02:19:26.004316Z"}}
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

# %% [code] {"execution":{"iopub.status.busy":"2022-02-24T02:19:28.076844Z","iopub.execute_input":"2022-02-24T02:19:28.077172Z","iopub.status.idle":"2022-02-24T02:19:28.464508Z","shell.execute_reply.started":"2022-02-24T02:19:28.077131Z","shell.execute_reply":"2022-02-24T02:19:28.463002Z"}}
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

# %% [code] {"execution":{"iopub.status.busy":"2022-02-24T02:19:30.718223Z","iopub.execute_input":"2022-02-24T02:19:30.718566Z","iopub.status.idle":"2022-02-24T02:52:37.328071Z","shell.execute_reply.started":"2022-02-24T02:19:30.718527Z","shell.execute_reply":"2022-02-24T02:52:37.326993Z"}}
start = time.time()
rlr = ReduceLROnPlateau(monitor='val_accuracy', verbose=1, mode='max',factor=0.3, min_lr=1e-6, patience=2)

history = model.fit(train, epochs=10, validation_data=val, verbose=1, callbacks=[rlr])
print("Total time: ", time.time() - start, "seconds")
model.evaluate(test)

# %% [code] {"execution":{"iopub.status.busy":"2022-02-24T02:53:03.585872Z","iopub.execute_input":"2022-02-24T02:53:03.586167Z","iopub.status.idle":"2022-02-24T02:53:03.594557Z","shell.execute_reply.started":"2022-02-24T02:53:03.586138Z","shell.execute_reply":"2022-02-24T02:53:03.59375Z"}}
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

# %% [code] {"execution":{"iopub.status.busy":"2022-02-24T02:53:04.41939Z","iopub.execute_input":"2022-02-24T02:53:04.419867Z","iopub.status.idle":"2022-02-24T02:53:04.870874Z","shell.execute_reply.started":"2022-02-24T02:53:04.419818Z","shell.execute_reply":"2022-02-24T02:53:04.869893Z"}}
plotHist(history)

# %% [code] {"execution":{"iopub.status.busy":"2022-02-24T02:53:06.743848Z","iopub.execute_input":"2022-02-24T02:53:06.744502Z","iopub.status.idle":"2022-02-24T02:53:24.018249Z","shell.execute_reply.started":"2022-02-24T02:53:06.744459Z","shell.execute_reply":"2022-02-24T02:53:24.017242Z"}}
y_pred = model.predict(test)
y_pred_max = np.argmax(y_pred, axis=1)

labels = (train.class_indices)
labels = dict((v,k) for k,v in labels.items())
y_pred = [labels[k] for k in y_pred_max]

y_test = testDf.targets

# %% [code] {"execution":{"iopub.status.busy":"2022-02-24T02:53:51.630284Z","iopub.execute_input":"2022-02-24T02:53:51.631205Z","iopub.status.idle":"2022-02-24T02:53:52.206877Z","shell.execute_reply.started":"2022-02-24T02:53:51.63116Z","shell.execute_reply":"2022-02-24T02:53:52.205751Z"}}
cf_matrix = confusion_matrix(y_true=y_test, y_pred=y_pred)
ax = sns.heatmap(cf_matrix, annot=True, cmap='Blues', fmt='g')
print("Accuracy of the Model:",accuracy_score(y_test, y_pred)*100,"%")
print(classification_report(y_test, y_pred))

ax.set_title('Brain Tumor CT Scans Confusion Matrix \n\n');
ax.set_xlabel('\nPredicted Tumor Present')
ax.set_ylabel('Actual Tumor Present ');

ax.xaxis.set_ticklabels(['False','True'])
ax.yaxis.set_ticklabels(['False','True'])

plt.show()

# %% [code]
