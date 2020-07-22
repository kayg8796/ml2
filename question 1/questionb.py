# -*- coding: utf-8 -*-
"""
Created on Thu Dec 26 18:48:38 2019

@author: Ngomba Litombe
"""

import keras
from keras.models import Sequential
from keras.layers import Dense , Flatten
from keras.utils import to_categorical


# Importing the dataset
dataset=keras.datasets.mnist
(xtrain,ytrain),(xtest,ytest)=dataset.load_data()


# Normalizing dataset
xtrain = xtrain/255.0
xtest = xtest/255.0

#reshaping the input matrix
#xtrain = xtrain.reshape([-1,784])
#xtest = xtest.reshape([-1,784])

#onehotencoding
ytest=to_categorical(ytest)
ytrain=to_categorical(ytrain)

def network(act_name,num): #act_name is the activation function name and num is the number of hidden layers
    #initializing the ann as a sequence of layers
    model=Sequential()
    #input layer
    model.add(Dense(units=32,activation=act_name,input_shape=(28,28)))
    model.add(Flatten())
    #hidden layers
    for i in range(num-2):
        model.add(Dense(units=32,activation=act_name))
    model.add(Dense(units=10,activation='softmax'))
    
    
    model.compile(optimizer=keras.optimizers.SGD(lr=0.01),loss='categorical_crossentropy',metrics=['accuracy'])
    #fitting training data
    model.fit(xtrain,ytrain,batch_size=100,epochs=10)
    #ypred=classifier.predict(xtest)
    #print(history.accur)
    return model.evaluate(xtest,ytest)

model7=network('tanh',5)
model8=network('tanh',20)
model9=network('tanh',40)
model1=network('sigmoid',5)
model2=network('sigmoid',20)
model3=network('sigmoid',40)
model4=network('relu',5)
model5=network('relu',20)
model6=network('relu',40)


print('the sigmoid activation function with 5 layers trained for 10 epochs has loss: {} and accuracy: {}\n'.format(model1[0],model1[1]))
print('the sigmoid activation function with 20 layers for 10 epochs has loss: {} and accuracy: {}\n'.format(model2[0],model2[1]))
print('the sigmoid activation function with 40 layers for 10 epochs has loss: {} and accuracy: {}\n'.format(model3[0],model3[1]))
print('the relu activation function with 5 layers for 10 epochs has loss: {} and accuracy: {}\n'.format(model4[0],model4[1]))
print('the relu activation function with 20 layers for 10 epochs has loss: {} and accuracy: {}\n'.format(model5[0],model5[1]))
print('the relu activation function with 40 layers for 10 epochs has loss: {} and accuracy: {}\n'.format(model6[0],model6[1]))
print('the tanh activation function with 5 layers for 10 epochs has loss: {} and accuracy: {}\n'.format(model7[0],model7[1]))
print('the tanh activation function with 20 layers for 10 epochs has loss: {} and accuracy: {}\n'.format(model8[0],model8[1]))
print('the tanh activation function with 40 layers for 10 epochs has loss: {} and accuracy: {}\n'.format(model9[0],model9[1]))