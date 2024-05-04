import tensorflow as tf
from keras.optimizers import RMSprop
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from keras.losses import SparseCategoricalCrossentropy


def create_model():

    model = tf.keras.models.Sequential([ 

        tf.keras.layers.Conv2D(16, (3,3), activation='relu', input_shape=(224, 224, 3), strides=(2,2)),
        tf.keras.layers.Conv2D(72, (1,1), strides=(1,1)),
        tf.keras.layers.Conv2D(32, (3,3), activation='relu'),
        tf.keras.layers.MaxPooling2D(2,2),
        tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
        tf.keras.layers.MaxPooling2D(2,2),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(512, activation='relu'),
        tf.keras.layers.Dense(7, activation='sigmoid')
    ])
    model.compile(loss=SparseCategoricalCrossentropy,
                    optimizer=RMSprop(learning_rate=1e-4),
                    metrics=['accuracy'])
    
    return model

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
        target_size=(224, 224), 
        batch_size=20,
        # Since we use binary_crossentropy loss, we need binary labels
        class_mode='sparse')

validation_datagen = ImageDataGenerator(rescale=1./255)


# Flow validation images in batches of 20 using test_datagen generator
validation_generator = validation_datagen.flow_from_directory(
        './DATASET/test/',
        target_size=(224, 224),
        batch_size=20,
        class_mode='sparse')

model = create_model()

model.summary()

history = model.fit(
      train_generator,
      epochs=100,
      validation_data = validation_generator)

