import keras
from keras import backend as K
from keras.layers import Dense, Activation, Dropout, Reshape, Softmax, Dot, Flatten
from keras.optimizers import Adam
from keras.metrics import categorical_crossentropy
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing import image
from keras.models import Model
from keras.applications import imagenet_utils
from keras.layers import Dense,GlobalAveragePooling2D
from keras.applications import MobileNet
from keras.applications.mobilenet import preprocess_input
import numpy as np
from IPython.display import Image
from keras.optimizers import Adam

base_model=MobileNet(weights='imagenet',include_top=False) #imports the mobilenet model and discards the last 1000 neuron layer.

x=base_model.output

# freeze all layers in the base model
base_model.trainable = True

# # un-freeze the BatchNorm layers
# for layer in base_model.layers:
#     if "BatchNormalization" in layer.__class__.__name__:
#         layer.trainable = True

x = GlobalAveragePooling2D()(x)
x = Dense(1024, activation='relu')(x)
x = Dropout(0.5)(x)
x = Dense(7, activation='softmax')(x)

model=Model(inputs=base_model.input,outputs=x)


train_datagen=ImageDataGenerator(preprocessing_function=preprocess_input) #included in our dependencies

train_generator=train_datagen.flow_from_directory('./DATASET/train',
                                                 target_size=(224,224),
                                                 color_mode='rgb',
                                                 batch_size=32,
                                                 class_mode='categorical',
                                                 shuffle=True)

validation_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)


# Flow validation images in batches of 20 using test_datagen generator
validation_generator = validation_datagen.flow_from_directory(
        './DATASET/test/',
        target_size=(224, 224),
        batch_size=32,
        class_mode='categorical')

model.compile(optimizer='Adam',loss='categorical_crossentropy',metrics=['accuracy'])

model.fit(train_generator,
                    validation_data=validation_generator,
                   epochs=100)