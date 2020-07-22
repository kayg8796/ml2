

import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.utils import to_categorical
from keras import backend as K
import numpy as np
import matplotlib.pyplot as plt


        
def Le_cun(x):
    lc=1.7159 * keras.activations.tanh((2/3)*x) + 0.01 *x
    return lc


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

def network(act_name,num,gradient=True): #act_name is the activation function name and num is the number of hidden layers
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
    #returning gradient values for untrained network
    if gradient:
        #ypred=classifier.predict(xtest)
        #print(history.accur)
        grads = get_gradients(classifier, xtrain, ytrain) #on a given mini batch , use a minibatch
        max_grads=[]
        k=0
        while(k<len(grads)):
            max_grads.append(np.max([np.max(grads[k]),np.max(grads[k+1])]))#taking the max val of weight and bias gradient of a layer
            k+=2
        return max_grads
    else:
        his=classifier.fit(xtrain,ytrain,batch_size=100,epochs=10, validation_data=(xtest,ytest))
        return his.history['accuracy']
        
    

grad_Lc5=network(Le_cun,5,True)
grad_Lc20=network(Le_cun,20,True)
grad_Lc40=network(Le_cun,40,True)
grad_tan5=network('tanh',5,True)
grad_tan20=network('tanh',20,True)
grad_tan40=network('tanh',40,True)

#x axis
x1 = np.linspace(1,5,5)
x2 = np.linspace(1,20,20)
x3 = np.linspace(1,40,40)

plt.plot(x1,grad_Lc5,label='lecun')
plt.plot(x1,grad_tan5,label='tanh')
plt.ylabel('gradients')
plt.xlabel('layer depth')
plt.legend()
plt.title('5 layers untrained')
plt.show()

plt.plot(x2,grad_Lc20,label='lecun')
plt.plot(x2,grad_tan20,label='tanh')
plt.legend()
plt.ylabel('gradients')
plt.xlabel('layer depth')
plt.title('20 layers untrained')
plt.show()

plt.plot(x3,grad_Lc40,label='lecun')
plt.plot(x3,grad_tan40,label='tanh')
plt.legend()
plt.ylabel('gradients')
plt.xlabel('layer depth')
plt.title('40 layers untrained')
plt.show()

#can easily write a loop that handles this

#Learning curve
accur_Lc5=network(Le_cun,5,False)
accur_Lc20=network(Le_cun,20,False)
accur_Lc40=network(Le_cun,40,False)
accur_tan5=network('tanh',5,False)
accur_tan20=network('tanh',20,False)
accur_tan40=network('tanh',40,False)

eps = np.arange(1,11)
plt.plot(eps,accur_Lc5,label='lecun')
plt.plot(eps,accur_tan5,label='tanh')
plt.legend()
plt.xlabel('epochs')
plt.ylabel('accuracy')
plt.title('Learning Currve for l=5')
plt.show()

plt.plot(eps,accur_Lc20,label='lecun')
plt.plot(eps,accur_tan20,label='tanh')
plt.legend()
plt.xlabel('epochs')
plt.ylabel('accuracy')
plt.title('Learning Curve for l=20')
plt.show()

plt.plot(eps,accur_Lc40,label='lecun')
plt.plot(eps,accur_tan40,label='tanh')
plt.legend()
plt.xlabel('epochs')
plt.ylabel('accuracy')
plt.title('Llearning curve for l=40')
plt.show()