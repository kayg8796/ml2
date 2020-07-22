# -*- coding: utf-8 -*-
"""
Created on Thu Dec 26 18:48:38 2019

@author: Ngomba Litombe
"""

import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.utils import to_categorical
from keras import backend as K
import numpy as np
import matplotlib.pyplot as plt


# Importing the dataset
dataset=keras.datasets.mnist
(xtrain,ytrain),(xtest,ytest)=dataset.load_data()


# Feature Scaling
xtrain = xtrain/255.0
xtest = xtest/255.0

#reshaping the input matrix
xtrain = xtrain.reshape([-1,784])
xtest = xtest.reshape([-1,784])

#onehotencoding
ytest=to_categorical(ytest)
ytrain=to_categorical(ytrain)

def get_gradients(model, inputs, outputs):
    grads = model.optimizer.get_gradients(model.total_loss, model.trainable_weights)
    symb_inputs = (model._feed_inputs + model._feed_targets + model._feed_sample_weights)
    f = K.function(symb_inputs, grads)
    x, y, weight = model._standardize_user_data(inputs, outputs)
    output_grad = f(x + y + weight)
    return np.array(output_grad)

def network(act_name,num): #act_name is the activation function name and num is the number of hidden layers
    #initializing the ann as a sequence of layers
    classifier=Sequential()
    #now adding layers
    #input layer
    classifier.add(Dense(units=32,activation=act_name,input_dim=(784)))
    #hidden layers
    for i in range(num-2):
        classifier.add(Dense(units=32,activation=act_name))
    classifier.add(Dense(units=10,activation='softmax'))
    
    
    classifier.compile(optimizer=keras.optimizers.SGD(lr=0.01),loss='categorical_crossentropy',metrics=['accuracy'])
    #fitting training data
    classifier.fit(xtrain,ytrain,batch_size=100,epochs=3)
    #ypred=classifier.predict(xtest)
    #print(history.accur)
    grads = get_gradients(classifier, xtrain, ytrain) #on a given mini batch , use a minibatch
    max_grads=[]
    k=0
    while(k<len(grads)):
        max_grads.append(np.max([np.max(grads[k]),np.max(grads[k+1])]))#taking the max val of weight and bias gradient of a layer
        k+=2
    return max_grads

grad_sig5=network('sigmoid',5)
grad_sig20=network('sigmoid',20)
grad_sig40=network('sigmoid',40)
grad_rel5=network('relu',5)
grad_rel20=network('relu',20)
grad_rel40=network('relu',40)
grad_tan5=network('tanh',5)
grad_tan20=network('tanh',20)
grad_tan40=network('tanh',40)

#x axis
x1 = np.linspace(1,5,5)
x2 = np.linspace(1,20,20)
x3 = np.linspace(1,40,40)

plt.plot(x1,grad_sig5,label='sigmoid')
plt.plot(x1,grad_rel5,label='relu')
plt.plot(x1,grad_tan5,label='tanh')
plt.legend()
plt.xlabel('layer depth')
plt.ylabel('max gradient')
plt.title('Network with 5 layers')
plt.show()

plt.plot(x2,grad_sig20,label='sigmoid')
plt.plot(x2,grad_rel20,label='relu')
plt.plot(x2,grad_tan20,label='tanh')
plt.legend()
plt.xlabel('layer depth')
plt.ylabel('max gradient')
plt.title('Network with 20 layers')
plt.show()

plt.plot(x3,grad_sig40,label='sigmoid')
plt.plot(x3,grad_rel40,label='relu')
plt.plot(x3,grad_tan40,label='tanh')
plt.legend()
plt.xlabel('layer depth')
plt.ylabel('max gradient')
plt.title('Network with 40 layers')
plt.show()

#can easily write a loop that handles this