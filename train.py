import tensorflow as tf
from tensorflow import keras 
from keras import layers, activations, Model
from keras.losses import SparseCategoricalCrossentropy
from keras.layers import Dense, Conv2D, BatchNormalization, Activation
from keras.layers import Add, ReLU, MaxPool2D, DepthwiseConv2D
from keras.models import Sequential, load_model
from keras.optimizers import Adam

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
            print(x)
        x = BatchNormalization()(x)
        x = Activation(activations.relu)(x)
        x = ConvBNReLU(x, filters, 1, strides=1)
        print(x)
        if i >= 1:
            inputs = Conv2D(filters=filters, kernel_size=(1,1), strides=1, padding='same')(inputs)
            x = Add()([x, inputs])
        inputs = x
    return x

input_img = keras.Input(shape=(56, 56, 64))  # Assuming image size of 32x32x3 (RGB)

out = bottleneckBlock(input_img, filters=64, kernel=3, e=2, s=2, reps=5)
out = Dense(10, activation='relu')(out)

model = Model(inputs=input_img, outputs=out)

model.compile(
    loss=SparseCategoricalCrossentropy(from_logits=False),
    optimizer=Adam(learning_rate=3e-4),
    metrics = ['accuracy'],
)   

model.summary()