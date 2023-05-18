import matplotlib.pyplot as plt
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import pandas as pd
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
from keras.utils import np_utils
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.optimizers import Adagrad
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.optimizers import Adadelta
from sklearn.model_selection import GridSearchCV
from keras.regularizers import l2
import time


def custom_activation(x):
    return x

"""Function scale uses the methods of normalization,centering and standardization to reduce big differences between values of the same feature"""
def scale(data_input,mode="Normalize"):
    cols = len(data_input[0,:])
    data_input = np.float64(data_input)
    if mode == "Normalize":
        for i in range(cols):
            m = max(data_input[:,i])
            if m != 0: data_input[:,i] = np.divide(data_input[:,i],m)
    elif mode == "Standardize":
        for i in range(cols):
            if data_input[:, i].std() != 0:
                data_input[:, i] = np.divide(np.subtract(data_input[:, i], data_input[:, i].mean()),data_input[:, i].std())
            else:
                data_input[:, i] = np.subtract(data_input[:, i], data_input[:, i].mean())
    elif mode == "Centering":
        for i in range(cols):
            data_input[:, i] = np.subtract(data_input[:, i], data_input[:, i].mean())
    else:
        print("The mode you have entered is invalid. Try using 'Normalize' or 'Standardize' or 'Centering'")
        exit(1)
    return data_input


model = Sequential()
model.add(Dense(9, input_dim=5, activation=custom_activation)) #,kernel_regularizer=l2(L2)
model.add(Dense(4, activation=custom_activation)) #,kernel_regularizer=l2(L2)
model.add(Dense(3, activation='softmax'))
optimizer = Adagrad(learning_rate=0.1) #,momentum=momentum
"""we have two model.compile below. The first uses CE loss function and the other uses MSE"""
model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

"""end"""
start = time.time() #begin timer

"""Pre-processing"""
trainDF = pd.read_csv("NoAccelNoCurrentAll.csv")
testDF = pd.read_csv("NoAccelNoCurrentTestAll.csv")

train_input = trainDF.iloc[:,1:].values
train_output = trainDF.iloc[:,0].values
test_input = testDF.iloc[:,1:].values
test_output = testDF.iloc[:,0].values

print("oxi")
train_input = scale(train_input,'Standardize')
test_input = scale(test_input,'Standardize')


trainEN = np_utils.to_categorical(train_output).astype(float) #Encode training output to vectors of 0 and 1
testEN = np_utils.to_categorical(test_output).astype(float) #Encode testing output to vectors of 0 and 1



"""end of pre-processing"""

"""Creating our architecture"""


ANN_model = model.fit(train_input, trainEN, epochs=25, batch_size=75, verbose=0)


end = time.time()

weight = model.get_weights()
np.savetxt('weight.csv', weight, fmt='%s', delimiter=',')

"""End of model"""

"""Plot accuracy - loss"""
history = ANN_model.history

plt.plot(history['accuracy'])
plt.plot(history['loss'])
plt.title('My model')
plt.ylabel('Accuracy - Loss')
plt.xlabel('epoch')
plt.legend(['accuracy', 'loss'], loc='upper left')

my_path = os.path.dirname(os.path.abspath(__file__))
plt.savefig(my_path + '\Results\TestSet18\epochs25batch75\Graph12.png')
plt.show()
"""End of plotting"""

"""Evaluate model in a completely new dataset"""

Accuracy = history['accuracy'][-1]

Prediction_Test = model.predict(test_input)
right = 0
for i in range(len(Prediction_Test)):
    temp = 0
    counter = 0
    for j in Prediction_Test[i]:
        if j > temp:
            temp = j
            prediction = counter
        counter += 1
    if test_output[i] == prediction:
        right += 1

Accuracy_Test = right / len(Prediction_Test)


"""End of Evaluation"""

"""Print results for training accuracy and hyperparameters used. Also display the timer."""

print("Accuracy of train set is %.2f"%(Accuracy*100) , "%" )

print("Accuracy of test set is %.2f"%(Accuracy_Test*100) , "%" )

print("Total time estimated: {0}".format(end-start))



"""End"""

