import tensorflow as tf
from tensorflow import keras 
from keras import layers, activations
from keras.layers import Dense, Conv2D, BatchNormalization, Activation
from keras.layers import Add, ReLU, MaxPool2D, DepthwiseConv2D
from keras.models import Sequential, load_model

def ConvBNReLU(x, filters, kernel_size, strides=1):
    x = Conv2D(filters=filters, kernel_size=kernel_size, strides=strides, padding='same')(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)

    return x

def identity_block(tensor, filters, stride):
    
    x = Conv2D(tensor, filters=filters, kernel_size=1, strides=stride)
    x = BatchNormalization()(x)
    
    x = Add()([tensor,x])    #skip connection
    x = ReLU()(x)
    
    return x

def bottleneckBlock(inputs, filters, kernel, e, s, reps):

    tchannel = int(e)

    for _ in reps:
        x = Conv2D(inputs, tchannel*filters, (1, 1), (1, 1), activation='RE')
        x = BatchNormalization()(x)

        x = DepthwiseConv2D(kernel, strides=(s, s), depth_multiplier=1, padding='same')(x)
        x = BatchNormalization()(x)
        x = Activation(activations.relu)(x)
        x = Add()([x, inputs])
    return x

