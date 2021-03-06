# -*- coding: utf-8 -*-
"""DL_Assignment2_Part_B.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1YJpT7jvk8zoSnJUN_K1jqBPStxO8W76k

\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#
#     PART B Section I       # 
\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#
"""

from google.colab import drive
drive.mount('/content/gdrive')

!unzip "/content/gdrive/My Drive/Colab Notebooks/DL Assignment 2/data1-2.h5.zip"

import tensorflow as tf
import numpy as np
import h5py
import matplotlib.pyplot as plt

# Function to load Images from data1.h5 
def loadDataH5():
  with h5py.File('data1.h5','r') as hf:
    trainX = np.array(hf.get('trainX'))
    trainY = np.array(hf.get('trainY'))
    valX = np.array(hf.get('valX'))
    valY = np.array(hf.get('valY'))
    print (trainX.shape,trainY.shape)
    print (valX.shape,valY.shape)
  return trainX, trainY, valX, valY

trainX, trainY, testX, testY = loadDataH5()

"""##1) CNN Model VGG16"""

import tensorflow as tf

# Importation of pre trained CNN VGG16
vggModel= tf.keras.applications.VGG16(weights='imagenet', include_top=False, input_shape=(128, 128, 3))
print (vggModel.summary())

# A prediction is made on training data
featuresTrain= vggModel.predict(trainX)

#reshape to flatten feature data
featuresTrain= featuresTrain.reshape(featuresTrain.shape[0], -1)

#  A prediction is made on test data
featuresVal = vggModel.predict(testX)

#reshape to flatten feature data
featuresVal= featuresVal.reshape(featuresVal.shape[0], -1)

### Section evaluating the different ML classifer

print("### CNN Model: VGG16 ")
print("  Accuracy Evaluation of different ML classifier")


## Random Forest
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
model = RandomForestClassifier(700)
model.fit(featuresTrain, trainY)

# evaluate the model
results = model.predict(featuresVal)
print ("     Random Forest:", metrics.accuracy_score(results, testY))



## Decision Tree
from sklearn.tree import DecisionTreeClassifier
model = DecisionTreeClassifier()
model.fit(featuresTrain, trainY)

# evaluate the model
results = model.predict(featuresVal)
print ("     Decision Tree:", metrics.accuracy_score(results, testY))



# KNN
from sklearn.neighbors import KNeighborsClassifier
model = KNeighborsClassifier(n_neighbors=9, p=3)
model.fit(featuresTrain, trainY)

# evaluate the model
results = model.predict(featuresVal)
print ("     KNN          :", metrics.accuracy_score(results, testY))


# Naive Bayes
from sklearn.naive_bayes import GaussianNB
model = GaussianNB()
model.fit(featuresTrain, trainY)

# evaluate the model
results = model.predict(featuresVal)
print ("     Naive Bayes  :", metrics.accuracy_score(results, testY))


# SVM
from sklearn.svm import SVC
model = SVC(gamma="auto")
model.fit(featuresTrain, trainY)

# evaluate the model
results = model.predict(featuresVal)
print ("     SVM          :", metrics.accuracy_score(results, testY))

"""##2) CNN Model InceptionV3"""

### CNN model: InceptionV3

# Importation of pre trained CNN InceptionV3
InceptionV3Model= tf.keras.applications.InceptionV3(weights='imagenet', include_top=False, input_shape=(128, 128, 3))
print (InceptionV3Model.summary())

# A prediction is made on training data
featuresTrain= InceptionV3Model.predict(trainX)

#reshape to flatten feature data
featuresTrain= featuresTrain.reshape(featuresTrain.shape[0], -1)

# A prediction is made on test data
featuresVal = InceptionV3Model.predict(testX)

#reshape to flatten feature data
featuresVal= featuresVal.reshape(featuresVal.shape[0], -1)

### Section evaluating the different ML classifer

print("### CNN Model: InceptionV3 ")
print("  Accuracy Evaluation of different ML classifier")


## Random Forest
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
model = RandomForestClassifier(700)
model.fit(featuresTrain, trainY)

# evaluate the model
results = model.predict(featuresVal)
print ("     Random Forest:", metrics.accuracy_score(results, testY))



## Decision Tree
from sklearn.tree import DecisionTreeClassifier
model = DecisionTreeClassifier()
model.fit(featuresTrain, trainY)

# evaluate the model
results = model.predict(featuresVal)
print ("     Decision Tree:", metrics.accuracy_score(results, testY))



# KNN
from sklearn.neighbors import KNeighborsClassifier
model = KNeighborsClassifier(n_neighbors=9, p=3)
model.fit(featuresTrain, trainY)

# evaluate the model
results = model.predict(featuresVal)
print ("     KNN          :", metrics.accuracy_score(results, testY))


# Naive Bayes
from sklearn.naive_bayes import GaussianNB
model = GaussianNB()
model.fit(featuresTrain, trainY)

# evaluate the model
results = model.predict(featuresVal)
print ("     Naive Bayes  :", metrics.accuracy_score(results, testY))


# SVM
from sklearn.svm import SVC
model = SVC(gamma="auto")
model.fit(featuresTrain, trainY)

# evaluate the model
results = model.predict(featuresVal)
print ("     SVM          :", metrics.accuracy_score(results, testY))

"""##3) CNN Model ResNet50"""

### CNN model: ResNet50

# Importation of pre trained CNN ResNet50
ResNet50Model= tf.keras.applications.ResNet50(weights='imagenet', include_top=False, input_shape=(128, 128, 3))
print (ResNet50Model.summary())

# A prediction is made on training data
featuresTrain= ResNet50Model.predict(trainX)

#reshape to flatten feature data
featuresTrain= featuresTrain.reshape(featuresTrain.shape[0], -1)

# A prediction is made on test data
featuresVal = ResNet50Model.predict(testX)

#reshape to flatten feature data
featuresVal= featuresVal.reshape(featuresVal.shape[0], -1)

### Section evaluating the different ML classifer

print("### CNN Model: ResNet50 ")
print("  Accuracy Evaluation of different ML classifier")


## Random Forest
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
model = RandomForestClassifier(700)
model.fit(featuresTrain, trainY)

# evaluate the model
results = model.predict(featuresVal)
print ("     Random Forest:", metrics.accuracy_score(results, testY))



## Decision Tree
from sklearn.tree import DecisionTreeClassifier
model = DecisionTreeClassifier()
model.fit(featuresTrain, trainY)

# evaluate the model
results = model.predict(featuresVal)
print ("     Decision Tree:", metrics.accuracy_score(results, testY))



# KNN
from sklearn.neighbors import KNeighborsClassifier
model = KNeighborsClassifier(n_neighbors=9, p=3)
model.fit(featuresTrain, trainY)

# evaluate the model
results = model.predict(featuresVal)
print ("     KNN          :", metrics.accuracy_score(results, testY))


# Naive Bayes
from sklearn.naive_bayes import GaussianNB
model = GaussianNB()
model.fit(featuresTrain, trainY)

# evaluate the model
results = model.predict(featuresVal)
print ("     Naive Bayes  :", metrics.accuracy_score(results, testY))


# SVM
from sklearn.svm import SVC
model = SVC(gamma="auto")
model.fit(featuresTrain, trainY)

# evaluate the model
results = model.predict(featuresVal)
print ("     SVM          :", metrics.accuracy_score(results, testY))

"""\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#
#     PART B Section II       # 
\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#\#

##1) Experiment1: CNN model VGG16 with SGD optimizer
"""

import tensorflow as tf

# Load the ImageNet VGG model. 
# extra conv layers, dropout and softmax activation is added 

vggModel= tf.keras.applications.VGG16( weights='imagenet', include_top=False, input_shape=(128, 128, 3))
vggModel.trainable= False
model = tf.keras.models.Sequential()
model.add(vggModel)
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(256, activation='relu'))
model.add(tf.keras.layers.Dropout(0.5))
model.add(tf.keras.layers.Dense(17, activation='softmax'))

print (model.summary())

# The built model is compiled with SGD optimizer
model.compile(loss='sparse_categorical_crossentropy', 
              optimizer=tf.keras.optimizers.SGD(lr=0.01),
              metrics=['accuracy'])

# The model is trained
H =model.fit(trainX, trainY, epochs=12, batch_size=51, validation_data=(testX, testY))

# Prediction on test data
result_predictions = model.predict(testX)
# select the class with the highest value
predictions = np.argmax(result_predictions, axis=1)
# Check if the model prediction is correct (True if prediction correct, False otherwise)
correct = np.equal(predictions, testY)
# Conversion of the boolean array into a numerical array (1 if True, 0 if False)
pred_correct = correct.astype(np.float32)
# mean value of predictions_correct
accuracy = np.mean(pred_correct)
print("CNN model VGG16, SGD optimizer accuracy:",accuracy)

"""##2) Experiment2: CNN model VGG16 with Nadam optimizer"""

# The model is now compiled with Nadam optimizer
model.compile(loss='sparse_categorical_crossentropy', 
              optimizer=tf.keras.optimizers.Nadam(lr=0.001),
              metrics=['accuracy'])

# The model is trained
H =model.fit(trainX, trainY, epochs=12, batch_size=51, validation_data=(testX, testY))

# Prediction on test data
result_predictions = model.predict(testX)
# select the class with the highest value
predictions = np.argmax(result_predictions, axis=1)
# Check if the model prediction is correct (True if prediction correct, False otherwise)
correct = np.equal(predictions, testY)
# Conversion of the boolean array into a numerical array (1 if True, 0 if False)
pred_correct = correct.astype(np.float32)
# mean value of predictions_correct
accuracy = np.mean(pred_correct)
print("CNN model VGG16, Nadam optimizer:",accuracy)

"""##3) Experiment3: VGG16 Fine-tuning from block4"""

import matplotlib.pyplot as plt
# plot the training loss and accuracy
plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0, 12), H.history["loss"], label="train_loss")
plt.plot(np.arange(0, 12), H.history["val_loss"], label="val_loss")
plt.plot(np.arange(0, 12), H.history["acc"], label="train_acc")
plt.plot(np.arange(0, 12), H.history["val_acc"], label="val_acc")
plt.title("Training Loss and Accuracy")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend()
plt.show()

### In this section we start with the fine-tuning

# the VGG16 is now unfrozen and is ready to be trained 
vggModel.trainable= True

# Initilize the trainable flag to false
trainableFlag= False

# a loop goes through each layer of VGG16
for layer in vggModel.layers:
  # It checks if the layer name is block4_conv1
  if layer.name== 'block4_conv1':
    # when the specified layer name is found, the trainable flag is set to True
    # This means that all layers from the specified layer could be trained
    trainableFlag= True
  layer.trainable= trainableFlag

# The model is compiled with the Nadam optimizer with a very low learning rate  
model.compile(loss='sparse_categorical_crossentropy',optimizer=tf.keras.optimizers.Nadam(lr=1e-5),metrics=['accuracy'])

# The model is trained
H =model.fit(trainX, trainY, epochs=60, batch_size=51, validation_data=(testX, testY))

import matplotlib.pyplot as plt
# plot the training loss and accuracy
plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0, 60), H.history["loss"], label="train_loss")
plt.plot(np.arange(0, 60), H.history["val_loss"], label="val_loss")
plt.plot(np.arange(0, 60), H.history["acc"], label="train_acc")
plt.plot(np.arange(0, 60), H.history["val_acc"], label="val_acc")
plt.title("Training Loss and Accuracy")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend()
plt.show()

"""##4) Experiment4: VGG16 Fine-tuning from block3"""

### In this section we start with the fine-tuning

# the VGG16 is now unfrozen and is ready to be trained 
vggModel.trainable= True

# Initilize the trainable flag to false
trainableFlag= False

# a loop goes through each layer of VGG16
for layer in vggModel.layers:
  # It checks if the layer name is block3_conv1
  if layer.name== 'block3_conv1':
    # when the specified layer name is found, the trainable flag is set to True
    # This means that all layers from the specified layer could be trained
    trainableFlag= True
  layer.trainable= trainableFlag

# The model is compiled with the Nadam optimizer with a very low learning rate  
model.compile(loss='sparse_categorical_crossentropy',optimizer=tf.keras.optimizers.Nadam(lr=1e-5),metrics=['accuracy'])

# The model is trained
H =model.fit(trainX, trainY, epochs=60, batch_size=51, validation_data=(testX, testY))

# Prediction on test data
result_predictions = model.predict(testX)
# select the class with the highest value
predictions = np.argmax(result_predictions, axis=1)
# Check if the model prediction is correct (True if prediction correct, False otherwise)
correct = np.equal(predictions, testY)
# Conversion of the boolean array into a numerical array (1 if True, 0 if False)
pred_correct = correct.astype(np.float32)
# mean value of predictions_correct
accuracy = np.mean(pred_correct)
print("CNN model VGG16, Nadam optimizer:")
print("  Fine-tuning using Nadam optimiser from block3_conv1")
print("  Accuracy after fine-tuning:", accuracy)

"""##5) Experiment5: CNN model InceptionV3 with Nadam optimizer"""

import tensorflow as tf

# Load the ImageNet VGG model. 
# extra conv layers, dropout and softmax activation is added 

InceptionV3Model= tf.keras.applications.InceptionV3(weights='imagenet', include_top=False, input_shape=(128, 128, 3))
InceptionV3Model.trainable= False
modelinc = tf.keras.models.Sequential()
modelinc.add(InceptionV3Model)
modelinc.add(tf.keras.layers.Flatten())
modelinc.add(tf.keras.layers.Dense(256, activation='relu'))
modelinc.add(tf.keras.layers.Dropout(0.5))
modelinc.add(tf.keras.layers.Dense(17, activation='softmax'))

print (modelinc.summary())

# The built model is compiled with Nadam optimizer
modelinc.compile(loss='sparse_categorical_crossentropy', 
              optimizer=tf.keras.optimizers.Nadam(lr=0.001),
              metrics=['accuracy'])

# The model is trained
H =modelinc.fit(trainX, trainY, epochs=50, batch_size=51, validation_data=(testX, testY))

import matplotlib.pyplot as plt
# plot the training loss and accuracy
plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0, 50), H.history["loss"], label="train_loss")
plt.plot(np.arange(0, 50), H.history["val_loss"], label="val_loss")
plt.plot(np.arange(0, 50), H.history["acc"], label="train_acc")
plt.plot(np.arange(0, 50), H.history["val_acc"], label="val_acc")
plt.title("Training Loss and Accuracy")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend()
plt.show()

# Prediction on test data
result_predictions = modelinc.predict(testX)
# select the class with the highest value
predictions = np.argmax(result_predictions, axis=1)
# Check if the model prediction is correct (True if prediction correct, False otherwise)
correct = np.equal(predictions, testY)
# Conversion of the boolean array into a numerical array (1 if True, 0 if False)
pred_correct = correct.astype(np.float32)
# mean value of predictions_correct
accuracy = np.mean(pred_correct)
print("CNN model InceptionV3, Nadam optimizer:",accuracy)