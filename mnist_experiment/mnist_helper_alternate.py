#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 17 10:42:27 2018

@author: shivap
"""

import os
import imp
import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Activation, Dense, Dropout, Flatten
from keras.layers import Conv2D, Convolution2D, MaxPooling2D
from keras import backend as K
#from tensorflow.contrib.keras import backend as K
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import matplotlib.pyplot as plt
from keras.models import model_from_json
from time import time

from keras.optimizers import SGD, Adam

# My other files for affine transform
import mnist_experiment.tf_shear_example as af
 
from sklearn.metrics import confusion_matrix

import numpy as np
#import cv2

batch_size = 128
#epochs = 12



#%%

# =============================================================================
# Display digit
# =============================================================================
def display_image(arr):    
    two_d = (np.reshape(arr, (28, 28)))
    plt.imshow(two_d, interpolation='nearest')
    return plt


# =============================================================================
# Plot gallery of MNIST images
# =============================================================================
def plot_gallery(images, h=28, w=28, n_row=3, n_col=4):
    """Helper function to plot a gallery of portraits"""
    plt.figure(figsize=(1.8 * n_col, 2.4 * n_row))
    plt.subplots_adjust(bottom=0, left=.01, right=.99, top=.90, hspace=.25)
    for i in range(len(images)):
        plt.subplot(n_row, n_col, i + 1)
        plt.imshow(images[i], cmap=plt.cm.gray)        
        plt.xticks(())
        plt.yticks(())

# =============================================================================
#  Function for general affine transform
# =============================================================================
def affine_transform(Xd, tmatrix, h=28, w=28):
    Xd = Xd.reshape(Xd.shape[0], 1, h, w) 
    transform_matrix = af.transform_matrix_offset_center(tmatrix, h, w)
    Xtransform = np.zeros(Xd.shape)
    for item in range(len(Xd)):
        Xtransform[item] = af.apply_transform(Xd[item], transform_matrix, channel_axis = 0, fill_mode='nearest', cval=0)
    Xtransform = Xtransform.reshape(Xtransform.shape[0], h, w) 
    return Xtransform 

# =============================================================================
# Rotate a given dataset
# =============================================================================
def applyRotation(Xd, angle):
    # Reshape data for our purpose
    h,w = Xd.shape[1:]
    Xd = Xd.reshape(Xd.shape[0], 1, h, w)     
    theta = np.pi / 180 * angle
    rotation_matrix = np.array([[np.cos(theta), -np.sin(theta), 0],
                              [np.sin(theta), np.cos(theta), 0], [0, 0, 1]])
    transform_matrix = af.transform_matrix_offset_center(rotation_matrix, h, w)
    Xrotate = np.zeros(Xd.shape)
    for item in range(len(Xd)):
        Xrotate[item] = af.apply_transform(Xd[item], transform_matrix, channel_axis = 0, fill_mode='nearest', cval=0)
    Xrotate = Xrotate.reshape(Xrotate.shape[0], h, w) 
    return Xrotate

# =============================================================================
# Skew a given dataset
# =============================================================================
def applyShear(Xd, x_shear=0, y_shear=0):
    h,w = Xd.shape[1:]     
    shear_matrix = np.array([[1, y_shear, 0], [x_shear,  1, 0],  [0, 0, 1]])    
    return affine_transform(Xd, shear_matrix, h, w)


# =============================================================================
# Function to train CNN for MNIST recognition
# =============================================================================
def trainCNN(x_train, y_train, filter_size, epochs):
    batch_size = 128 
    num_classes = len(np.unique(y_train))
    #epochs = 12
    # input image dimensions
#    img_rows, img_cols = 28, 28
    img_rows, img_cols = x_train.shape[1:]
    
    if K.image_data_format() == 'channels_first':
        x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)        
        input_shape = (1, img_rows, img_cols)
    else:
        x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)        
        input_shape = (img_rows, img_cols, 1)

    x_train = x_train.astype('float32')    
    x_train /= 255    
    print('x_train shape:', x_train.shape)
    print(x_train.shape[0], 'train samples')
    
    # convert class vectors to binary class matrices
    y_train = keras.utils.np_utils.to_categorical(y_train, num_classes)
    
    model = Sequential()
    model.add(Conv2D(32, kernel_size=(filter_size, filter_size),
                 activation='relu',
                 input_shape=input_shape))
    model.add(Conv2D(64, (filter_size, filter_size), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation='softmax'))

    model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adadelta(),
              metrics=['accuracy'])
    tic = time()
    model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          verbose=1)
    toc = time()
    print("Model trained in : ", (toc - tic)/(3600), " hrs")
    return model   


# =============================================================================
# Function to train CNN for MNIST recognition
# =============================================================================
def trainLargerCNN(x_train, y_train, filter_size=5, epochs=15):
    batch_size = 128 
    num_classes = len(np.unique(y_train))
    #epochs = 12
    # input image dimensions
#    img_rows, img_cols = 28, 28
    img_rows, img_cols = x_train.shape[1:]
    
    if K.image_data_format() == 'channels_first':
        x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)        
        input_shape = (1, img_rows, img_cols)
    else:
        x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)        
        input_shape = (img_rows, img_cols, 1)

    x_train = x_train.astype('float32')    
    x_train /= 255    
    print('x_train shape:', x_train.shape)
    print(x_train.shape[0], 'train samples')
    
    # convert class vectors to binary class matrices
    y_train = keras.utils.np_utils.to_categorical(y_train, num_classes)
    model = Sequential()
    model.add(Conv2D(30, (5, 5), input_shape=input_shape, activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(15, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.2))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dense(50, activation='relu'))
    model.add(Dense(num_classes, activation='softmax'))
    # Compile model
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    
    tic = time()
    history = model.fit(x_train, y_train, validation_split=0.2,
          batch_size=batch_size,
          epochs=epochs,
          verbose=1)
    toc = time()
    print("Model trained in : ", (toc - tic)/(3600), " hrs")
    return model, history



def PlotHistory(train_value, test_value, value_is_loss_or_acc):
    f, ax = plt.subplots()
    ax.plot([None] + train_value, 'o-')
    ax.plot([None] + test_value, 'x-')
    # Plot legend and use the best location automatically: loc = 0.
    ax.legend(['Train ' + value_is_loss_or_acc, 'Validation ' + value_is_loss_or_acc], loc = 0) 
    ax.set_title('Training/Validation ' + value_is_loss_or_acc + ' per Epoch')
    ax.set_xlabel('Epoch')
    ax.set_ylabel(value_is_loss_or_acc)  
#    plt.savefig("figs/Model_TrainValLoss.pdf")



# =============================================================================
# Function to LENET5 CNN for MNIST recognition
# =============================================================================
def trainLENETCNN(x_train, y_train, epochs=20, batch=256, num_filters=16, filter_size=5):
    batch_size = batch
    num_classes = len(np.unique(y_train))
    #epochs = 12
    # input image dimensions
#    img_rows, img_cols = 28, 28
    img_rows, img_cols = x_train.shape[1:]
    
    if K.image_data_format() == 'channels_first':
        x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)        
        input_shape = (1, img_rows, img_cols)
    else:
        x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)        
        input_shape = (img_rows, img_cols, 1)

    x_train = x_train.astype('float32')    
    x_train /= 255    
    print('x_train shape:', x_train.shape)
    print(x_train.shape[0], 'train samples')
    
    # convert class vectors to binary class matrices
    y_train = keras.utils.to_categorical(y_train, num_classes)
    
    # ---- LENET-5 Architecture ----
    model = Sequential()
    #C1 Layer
    model.add(Convolution2D(32, filter_size, filter_size, border_mode='same', input_shape=input_shape))
    # The activation for layers is ReLU
    model.add(Activation('relu'))
    # Max pooling
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    #Second Layer
    model.add(Convolution2D(num_filters, filter_size, filter_size, border_mode='valid', input_shape=(14,14,1)))
    model.add(Activation('relu'))
    # Max pooling
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    #Third Layer
    model.add(Convolution2D(num_filters, filter_size, filter_size, border_mode='valid', input_shape=(5,5,1)))
    #Flatten the CNN output
    model.add(Flatten())
    
    #Add Three dense Layer for the FNN     
    model.add(Dense(84))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    #model.add(Dense(32))
    #model.add(Activation('relu'))
    #model.add(Dropout(0.2))
    model.add(Dense(num_classes))
    # For classification, the activation is softmax
    model.add(Activation('softmax'))
    # Define optimizer
    #optmzr = SGD(lr=0.1, clipnorm=5.)
    optmzr = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
    # Define loss function = cross entropy
#    model.compile(loss='categorical_crossentropy', optimizer='adadelta', metrics=["accuracy"])
    model.compile(loss='categorical_crossentropy', optimizer=optmzr, metrics=["accuracy"])
    # ---- LENET-5 Architecture ----

    tic = time()
    train_history = model.fit(x_train, y_train,validation_split=0.2,
          batch_size=batch_size,
          epochs=epochs,
          verbose=1)
    toc = time()
    print("Model trained in : ", (toc - tic)/(3600), " hrs")
    return model, train_history   

# =============================================================================
# Testing prediction on test data modified by a transform
# =============================================================================
def test_transformMNIST(model, data, transform, measure):              
    x_test, y_test = data
    num_classes = len(np.unique(y_test))
    img_rows, img_cols = x_test.shape[1:]
    
    isRotate, isZoom, isShear = transform
    rotate_scale, x_shear, y_shear = measure
    
    if isShear:
        x_test = applyShear(x_test, x_shear, y_shear)
    if isRotate:
        x_test = applyRotation(x_test, rotate_scale)  
    
    if K.image_data_format() == 'channels_first':        
        x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
    else:        
        x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1) 
    x_test = x_test.astype('float32')    
    x_test /= 255
    
    # convert class vectors to binary class matrices    
    y_test = keras.utils.to_categorical(y_test, num_classes)    
    score = model.evaluate(x_test, y_test, verbose=0)
    return (1 - score[1])  

# =============================================================================
# Get error rate per digit
# =============================================================================
def test_ErrorDigittransformMNIST(model, data, transform, measure):    
    x_test, y_test = data
    num_classes = np.unique(y_test)
    img_rows, img_cols = x_test.shape[1:]
    
    isRotate, isZoom, isShear = transform
    rotate_scale, x_shear, y_shear = measure
    
    if isShear:
        x_test = applyShear(x_test, x_shear, y_shear)
    if isRotate:
        x_test = applyRotation(x_test, rotate_scale) 

    
    if K.image_data_format() == 'channels_first':        
        x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
    else:        
        x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1) 
    x_test = x_test.astype('float32')    
    x_test /= 255
    
    y_pred = model.predict_classes(x_test, verbose=0)
    cmat = confusion_matrix(y_test, y_pred, labels=num_classes)
    class_error_rate = 1 - cmat.diagonal()/cmat.sum(axis=1)
    return class_error_rate


# =============================================================================
# Function to Pad a given vector with value
# =============================================================================
def pad_with(vector, pad_width, iaxis, kwargs):
     pad_value = kwargs.get('padder', 10)
     vector[:pad_width[0]] = pad_value
     vector[-pad_width[1]:] = pad_value
     return vector


# =============================================================================
# Function to return padded MNIST. Padlength is the length of zeros to 
# be added on each side
# =============================================================================    
def padMNIST(pad_length):
    (x_train, y_train), (x_test, y_test) = mnist.load_data() 
    width = int(28 + pad_length*2)
    pXtrain = np.zeros((60000,width,width))
    pXtest  = np.zeros((10000,width,width))    
    for item in range(len(x_train)):
        pXtrain[item] = np.pad(x_train[item], pad_length, pad_with, padder=0)
    for item in range(len(x_test)):
        pXtest[item] = np.pad(x_test[item], pad_length, pad_with, padder=0)
    return pXtrain,y_train, pXtest,y_test     


# =============================================================================
#  Function to save the keras model
#   modelname should include the path also
# =============================================================================
def saveModel(model, modelname):
    # serialize model to JSON
    model_json = model.to_json()
    with open(modelname + ".json", "w") as json_file:
        json_file.write(model_json)
    # serialize weights to HDF5
    model.save_weights(modelname + ".h5")
    print("Saved model to disk")
    return

# =============================================================================
#  Function to load a keras model from a file
# =============================================================================
def loadModel(model_file):
    json_file = open(model_file + '.json' , 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    
    # load weights into new model
    loaded_model.load_weights(model_file + '.h5')
    print("Loaded model from disk")

    loaded_model.compile(loss=keras.losses.categorical_crossentropy, 
                     optimizer=keras.optimizers.Adadelta(), 
                     metrics=['accuracy'])
    model = loaded_model
    return model

    
    
    



