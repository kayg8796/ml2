# -*- coding: utf-8 -*-
"""
Created on Sun Jan  5 01:02:56 2020

@author: Ngomba Litombe
"""
# tweak plot function to get a better view
# randomize image selection for testing
# network libraries
from keras.models import Sequential
from keras.models import Model
from keras.layers import Dense
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.utils import to_categorical as cat
from keras.callbacks import EarlyStopping as stoppoint
import random

# loading a .mat file
from scipy.io import loadmat
import numpy as np
import matplotlib.pyplot as plt


# function for plotting the images
def plot_images(img, labels, nrows, ncols):
    """ Plot nrows x ncols images
    """
    fig, axes = plt.subplots(nrows, ncols)
    for i, ax in enumerate(axes.flat):
        if img[i].shape == (32, 32, 3):
            ax.imshow(img[i])
        else:
            ax.imshow(img[i, :, :, 0])
        ax.set_xticks([]);
        ax.set_yticks([])
        ax.set_title(labels[i])


dataset = loadmat('train_32x32.mat')
xtrain = dataset['X']
ytrain = dataset['y']
datatest = loadmat('test_32x32.mat')



xtest = datatest['X']
ytest = datatest['y']

xtrain = np.moveaxis(xtrain, -1, 0)
xtest = np.moveaxis(xtest, -1, 0)

# plotting sample images
# plot_images(xtrain,ytrain,2,2)

ytrain[ytrain == 10] = 0
ytrain = cat(ytrain.reshape([-1, 1]))

ytest[ytest == 10] = 0
ytest = cat(ytest.reshape([-1, 1]))

# initialising the CNN
classifier = Sequential()

# step 1 Convolution and maxpool layer 1
classifier.add(Convolution2D(9, (3, 3), input_shape=(32, 32, 3), padding='same', activation='relu'))
classifier.add(MaxPooling2D(pool_size=(3, 3)))

# adding a second convolutional and maxpool
classifier.add(Convolution2D(36, (3, 3), padding='same', activation='relu'))
classifier.add(MaxPooling2D(pool_size=(3, 3)))


# adding a second convolutional and maxpool
classifier.add(Convolution2D(49, (3, 3), padding='same', activation='relu'))
classifier.add(MaxPooling2D(pool_size=(3, 3)))


# step 3 : flattening
classifier.add(Flatten())

# multilabel logistic regression consisting of just the input layer then the output layer
# classifier.add(Dense(output_dim=64, activation='relu'))#what number of neurons should be used here?
classifier.add(Dense(output_dim=10, activation='softmax'))

classifier.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
cb = stoppoint(monitor='val_loss', patience=0)  # including earlystopping callback wtih
#  a patience of zero which means model stops training when the validation loss starts increases
classifier.fit(xtrain, ytrain, batch_size=100, callbacks=[cb], validation_split=7326 / len(xtrain), epochs=3)
ypred = classifier.predict(xtest)  # batch size is how much of the data we feed into the need at a time
# before readjusting the weights


# plotting confusion matrix
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

plt.figure(figsize=(12, 8))
cm = confusion_matrix(np.argmax(ytest, axis=1), np.argmax(ypred, axis=1))
# cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100.0 #normalizing the cm. add or might not
sns.heatmap(cm, annot=True, cmap='Reds', fmt='.1f', square=True)  # visualization

# plotting random image
# selecting the first image from the training set
plt.imshow(xtrain[2])

# i am going to work with the value two for now just to test how this goes
# later i would randomize the the image selection to pick one image from each class.
# let begin
# since we have nine layers, i would extract the 3 which are actually convolutional layers

conv_layers = [classifier.layers[i] for i in (0, 2, 4)]
layer_outputs = [layer.output for layer in conv_layers]
activation_model = Model(inputs=classifier.input, outputs=layer_outputs)


# creating a keras model functional api which is able to handle this layers indivually


def plot_conv_layers(image_1):
    activations = activation_model.predict(np.expand_dims(image_1, axis=0))
    # this produces an array of size 3. for instance array[0] has 9 different channels for the nine
    # different filters used ,

    # visualization of the images produce at each convolutional layer
    layer_names = []
    for layer in conv_layers:
        layer_names.append(layer.name)  # Names of the layers, so you can have them as part of your plot

    images_per_row = 9
    for layer_name, layer_activation in zip(layer_names, activations):  # Displays the feature maps
        n_features = layer_activation.shape[-1]  # Number of features in the feature map
        size = layer_activation.shape[1]  # The feature map has shape (1, size, size, n_features).
        n_cols = n_features // images_per_row  # Tiles the activation channels in this matrix
        display_grid = np.zeros((size * n_cols, images_per_row * size))
        for col in range(n_cols):  # Tiles each filter into a big horizontal grid
            for row in range(images_per_row):
                channel_image = layer_activation[0,
                                :, :,
                                col * images_per_row + row]
                channel_image -= channel_image.mean()  # Post-processes the feature to make it visually palatable
                channel_image /= channel_image.std()
                channel_image *= 64
                channel_image += 128
                channel_image = np.clip(channel_image, 0, 255).astype('uint8')
                display_grid[col * size: (col + 1) * size,  # Displays the grid
                row * size: (row + 1) * size] = channel_image
        scale = 1. / size
        plt.figure(figsize=(scale * display_grid.shape[1],
                            scale * display_grid.shape[0]))
        plt.title(layer_name)
        plt.grid(False)
        plt.imshow(display_grid, aspect='auto', cmap='viridis')


''' possible interpretation 
The first layer acts is arguably retaining the full shape of the digit, although there are several
 filters that are not activated and are left blank. At that stage, the activations retain almost all of 
 the information present in the initial picture.
As we go deeper in the layers, the activations become increasingly abstract and less visually
 interpretable. They begin to encode higher-level concepts such as single borders, corners and angles.
 Higher presentations carry increasingly less information about the visual contents of the image, and 
 increasingly more information related to the class of the image.

'''

# selecting random numbers for different classes
class_index = []
for c in np.unique(np.argmax(ytrain, axis=1)):
    class_index.append(np.where(np.argmax(ytrain, axis=1) == c))

# plotting convolutional layers for five random images selected from each class
for i in range(10):
    plot_conv_layers(xtrain[random.choice(class_index[i][0])])  # for digit 10 digits
