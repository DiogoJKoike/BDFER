import numpy as np
import os
import PIL
import PIL.Image
import tensorflow as tf
from keras.utils import image_dataset_from_directory
import pathlib

data_dir_test = pathlib.Path('./DATASET/test')
data_dir_train = pathlib.Path('./DATASET/train')

batch_size = 32
img_height = 224
img_width = 224

train_ds = image_dataset_from_directory(
  data_dir_train,
  validation_split=0.2,
  subset="training",
  seed=123,
  image_size=(img_height, img_width),
  batch_size=batch_size
)

test_ds = image_dataset_from_directory(
  data_dir_test,
  validation_split=0.2,
  subset="validation",
  seed=123,
  image_size=(img_height, img_width),
  batch_size=batch_size
)

class_names = train_ds.class_names
print(class_names)
class_names = test_ds.class_names
print(class_names)


import matplotlib.pyplot as plt

plt.figure(figsize=(10, 10))
for images, labels in train_ds.take(1):
  print(labels)
  for i in range(9):
    ax = plt.subplot(3, 3, i + 1)
    plt.imshow(images[i].numpy().astype("uint8"))
    plt.title(class_names[labels[i]])
    plt.axis("off")
plt.show()