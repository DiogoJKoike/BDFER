import tensorflow as tf
from keras import Model
from keras.layers import *
from keras.losses import *
from keras.activations import *
from keras.optimizers import *
from tensorflow.keras.preprocessing.image import ImageDataGenerator

def ConvBNReLU(inputs, kernel=3, filters=1, stride=1, groups=1):
    x = Conv2D(filters=filters, kernel_size=kernel, strides=stride, padding='same')(inputs)
    x = BatchNormalization()(x)
    x = ReLU(max_value=6)(x)

    return x

def DWSConv(inputs, filters, kernel, padding):
    x = DepthwiseConv2D(kernel_size=kernel, padding='same')(inputs)
    x = BatchNormalization()(x)
    x = ReLU()(x)

    x = Conv2D(filters=filters, kernel_size=(1,1), padding='same')(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)

    return x

def GDConv(inputs, kernel):
    x = DepthwiseConv2D(kernel_size=kernel)(inputs)
    x = BatchNormalization()(x)

    return x

def bottleneckBlock(inputs, filters, kernel, e, s, reps):

    for i in range(reps):
        stride = s if i == 0 else 1
        x = ConvBNReLU(inputs, kernel=1, filters=e*filters, stride=1, groups=e*filters)
        x = DepthwiseConv2D(kernel_size=3, padding='same', strides=stride)(x)
        x = Conv2D(kernel_size=1, filters=filters, strides=1)(x)
        x = BatchNormalization()(x)

        if s == 1:
            inputs = Conv2D(filters=filters, kernel_size=(1,1), strides=1, padding='same')(inputs)
            x = Add()([x, inputs])
        inputs = x
    return x


def scheduler(epoch, lr):
    if lr < 1e-7:
        return lr
    elif epoch%25 == 0 and epoch != 0:
        return lr * 0.5
    else:
        return lr


inputs = Input(shape=(112,112,3))

model = ConvBNReLU(inputs, filters=3, stride=2)
model = DWSConv(model, filters=64, kernel=3, padding=1)
model = bottleneckBlock(model, filters=64, kernel=3, e=2, s=2, reps=5)
model = bottleneckBlock(model, filters=128, kernel=3, e=4, s=2, reps=1)
model = bottleneckBlock(model, filters=128, kernel=3, e=2, s=1, reps=6)
model = bottleneckBlock(model, filters=128, kernel=3, e=4, s=2, reps=1)
model = bottleneckBlock(model, filters=128, kernel=3, e=2, s=1, reps=2)
model = ConvBNReLU(model, kernel=1, filters=512)
attentionx = GDConv(model, (7,1))(model)
attentiony = GDConv(model, (1,7))(model)

model = GDConv(model, kernel=7)
model = Conv2D(filters=128, kernel_size=1)(model)
model = BatchNormalization()(model)
model = Conv2D(7, (1, 1), padding='same', activation='softmax')(model)
model = Reshape((7,))(model)

model = Model(inputs=inputs, outputs=model)

model.summary()

train_datagen=ImageDataGenerator(
        rescale=1. / 255,
        shear_range=0.2,
        zoom_range=0.2,
        rotation_range=90,
        width_shift_range=0.2,
        height_shift_range=0.2,
        horizontal_flip=True)

train_generator=train_datagen.flow_from_directory('./DATASET/train',
                                                 target_size=(112,112),
                                                 color_mode='rgb',
                                                 batch_size=32,
                                                 class_mode='categorical',
                                                 shuffle=True)

validation_datagen = ImageDataGenerator(rescale=1. / 255)


# Flow validation images in batches of 20 using test_datagen generator
validation_generator = validation_datagen.flow_from_directory(
        './DATASET/test/',
        target_size=(112, 112),
        batch_size=32,
        class_mode='categorical')

opt = Adam(learning_rate=1e-4)

lr_callback = tf.keras.callbacks.LearningRateScheduler(scheduler, verbose=True)

model.compile(optimizer=opt,loss='categorical_crossentropy',metrics=['accuracy'])

model.fit(train_generator,
        validation_data=validation_generator,
        epochs=200,
        callbacks=[lr_callback])