import tensorflow as tf
from tensorflow import keras 
from keras import *
from keras.losses import SparseCategoricalCrossentropy
from keras.layers import Dense, Conv2D, BatchNormalization, Activation, GlobalAveragePooling2D
from keras.layers import Conv2D, MaxPool2D, Flatten,Dense,Dropout,BatchNormalization
from keras.models import Sequential, load_model
from keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
import os
import PIL
import PIL.Image
import tensorflow as tf
from keras.utils import image_dataset_from_directory
import pathlib

def ConvBNReLU(x, filters, kernel_size, strides=1):
    x = Conv2D(filters=filters, kernel_size=kernel_size, strides=strides, padding='same')(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)

    return x

# def identity_block(tensor, filters, stride):
    
#     x = Conv2D(tensor, filters=filters, kernel_size=1, strides=stride)
#     x = BatchNormalization()(x)
    
#     x = Add()([tensor,x])    #skip connection
#     x = ReLU()(x)
    
#     return x

def bottleneckBlock(inputs, filters, kernel, e, s, reps):

    for i in range(reps):
        x = ConvBNReLU(inputs, e*filters, 1)
        if i >= 1:
            x = DepthwiseConv2D(kernel, strides=(1, 1), depth_multiplier=1, padding='same')(x)
        else:
            x = DepthwiseConv2D(kernel, strides=(s, s), depth_multiplier=1, padding='same')(x)
        x = BatchNormalization()(x)
        x = Activation(activations.relu)(x)
        x = ConvBNReLU(x, filters, 1, strides=1)
        x = Dropout
        if i >= 1:
            inputs = Conv2D(filters=filters, kernel_size=(1,1), strides=1, padding='same')(inputs)
            x = Add()([x, inputs])
        inputs = x
    return x

data_dir_test = pathlib.Path('./DATASET/test')
data_dir_train = pathlib.Path('./DATASET/train')

train_datagen = ImageDataGenerator(
      rescale=1./255,
      rotation_range=40,
      width_shift_range=0.2,
      height_shift_range=0.2,
      shear_range=0.2,
      zoom_range=0.2,
      horizontal_flip=True,
      fill_mode='nearest')

# Flow training images in batches of 20 using train_datagen generator
train_generator = train_datagen.flow_from_directory(
        './DATASET/train/',  # This is the source directory for training images
        target_size=(100, 100), 
        batch_size=96,
        # Since we use binary_crossentropy loss, we need binary labels
        class_mode='categorical')

validation_datagen = ImageDataGenerator(rescale=1./255)


# Flow validation images in batches of 20 using test_datagen generator
validation_generator = validation_datagen.flow_from_directory(
        './DATASET/test/',
        target_size=(100, 100),
        batch_size=20,
        class_mode='categorical')

input_img = keras.Input(shape=(100, 100, 3))  

model= tf.keras.models.Sequential()
model.add(Conv2D(32, kernel_size=(3, 3), padding='same', activation='relu', input_shape=(100,100,3)))
model.add(Conv2D(64,(3,3), padding='same', activation='relu' ))
model.add(BatchNormalization())
model.add(MaxPool2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(128,(5,5), padding='same', activation='relu'))
model.add(BatchNormalization())
model.add(MaxPool2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
    
model.add(Conv2D(512,(3,3), padding='same', activation='relu', kernel_regularizer=regularizers.l2(0.01)))
model.add(BatchNormalization())
model.add(MaxPool2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(512,(3,3), padding='same', activation='relu', kernel_regularizer=regularizers.l2(0.01)))
model.add(BatchNormalization())
model.add(MaxPool2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Flatten()) 
model.add(Dense(256,activation = 'relu'))
model.add(BatchNormalization())
model.add(Dropout(0.25))
    
model.add(Dense(512,activation = 'relu'))
model.add(BatchNormalization())
model.add(Dropout(0.25))

model.add(Dense(7, activation='softmax'))

model.compile(
    loss='categorical_crossentropy',
    optimizer=Adam(learning_rate=1e-4),
    metrics = ['accuracy'],
)   

model.summary()

model.fit(
    train_generator,
    validation_data=validation_generator,
    epochs=100
)

