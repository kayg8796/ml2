#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import keras
from keras.models import Sequential
from keras.layers import Dense,Dropout
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.callbacks import EarlyStopping as stoppoint

#initialising the CNN
model = Sequential()

#step 1 Convolution
model.add(Conv2D(32,(3,3),input_shape=(28,28,1),activation='relu'))

#step 2 max polling
model.add(MaxPooling2D(pool_size=(2,2),padding = 'same'))

#adding a second convolutional layer
model.add(Conv2D(32,(3,3),activation='relu'))
model.add(MaxPooling2D(pool_size = (2, 2),padding = 'same'))

#step 3 : flattening
model.add(Flatten())
model.add(Dropout(0.1))
#step 4: full connection with the ann
model.add(Dense(units=128, activation='relu'))
model.add(Dense(units=10, activation='softmax'))

#step 5: compiling the CNN
model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])

model.summary()

#part 2: fitting the dataset to the CNN
from keras.preprocessing.image import ImageDataGenerator

#train_datagen was used alongside different combination of the parameters was used to perform dataset augmentation
#apriori to training the model
'''train_datagen = ImageDataGenerator(rescale = 1./255,
                                   shear_range = 0.2,
                                   zoom_range = [0.5,1.0],
                                   width_shift_range = [-5,5],
                                   vertical_flip = True)
train_datagen.fit(xtrain)
training_set = train_datagen.flow_from_directory('combined_data',
                                                 target_size = (28, 28),
                                                 color_mode = 'grayscale',
                                                 save_to_dir = 'dataset3',
                                                 batch_size = 10,
                                                 class_mode = 'categorical') '''

test_datagen = ImageDataGenerator(rescale = 1./255)
test_set = test_datagen.flow_from_directory('combined_data',
                                            target_size = (28, 28),
                                            color_mode='grayscale',
                                            batch_size = 10,
                                            class_mode = 'categorical',
                                            shuffle=True)
model.fit_generator(test_set,
                         steps_per_epoch = 1000, # number of batches to draw from the generator at each epoch
                         epochs = 10)


# In[ ]:


#import test data
import numpy as np
import cv2 as cv
from skimage.transform import resize
from keras.utils import to_categorical


# In[ ]:


#importing testset from directory

import numpy as np
import os
image_arr=[]
yt = np.array([])
data = os.listdir('dataset2')
for i in data:
    photos=os.listdir('dataset2/'+i)
    yt=np.concatenate((yt,int(i)*np.ones(len(photos))))
    for photo in photos:
        image_dir = 'dataset2/'+i+'/'+photo
        # load & smoothen image
        kernel = np.ones((7,7),np.float32)/49
        image = cv.imread(image_dir,cv.IMREAD_GRAYSCALE)
        #image = cv.filter2D(image,3,kernel)

        # make numpy array
        image = np.array(image)
        image = cv.resize(image, (28,28))
        # make negative
        #image = np.ones(image.shape) - image
        image_arr.append(image)
xt = np.array(image_arr)
yt=  [int(i) for i in yt]
yt = np.array(yt)

xt = xt/255.0
yt = yt.reshape([-1,1])
yt = to_categorical(yt)
xtr=np.expand_dims(xt, axis=0)
xtr = np.moveaxis(xtr,0,-1)


# In[ ]:


model.evaluate(xtr,yt)


# In[ ]:


(xtrain,ytrain) , (xtest,ytest) = keras.datasets.mnist.load_data()
xtest=np.expand_dims(xtest,axis=-1)
ytest = to_categorical(ytest)
xtrain=np.expand_dims(xtrain,axis=-1)
ytrain = to_categorical(ytrain)


# In[ ]:


model.evaluate(xtest,ytest)


# In[ ]:


#saving model
model.save('question3_model.h5')
print('model save to disk as : "question3_model.h5" \n')
#how to load model shown below
#model_new = keras.models.load_model('question3.h5')


# In[ ]:




