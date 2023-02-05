# import modules and packages to implement the code
import time

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# data processing, CSV file I/O (e.g. pd.read_csv)
import tensorflow as tf
from tensorflow import keras
import os
# from tensorflow.keras.preprocessing import image

import warnings
sns.set()
warnings.filterwarnings("ignore")

# import sklearn


classes = ['NORMAL', 'COVID-19']
# load the images from the directory ../data/non-enhanced
non_enhanced = os.listdir('../data/enhanced')

# define parameter for resnet50
IMG_SIZE = 224
BATCH_SIZE = 32
EPOCHS = 10

train_ds = tf.keras.preprocessing.image_dataset_from_directory(
    '../data/enhanced',
    validation_split=0.2,
    subset="training",
    seed=123,
    image_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE)

val_ds = tf.keras.preprocessing.image_dataset_from_directory(
    '../data/enhanced',
    validation_split=0.2,
    subset="validation",
    seed=123,
    image_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE)

class_names = train_ds.class_names
print(class_names)

# configure the dataset for performance
AUTOTUNE = tf.data.AUTOTUNE

train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

# standardize the data
normalization_layer = tf.keras.layers.experimental.preprocessing.Rescaling(1. / 255)

# resize the data
data_augmentation = keras.Sequential(
    [
        keras.layers.experimental.preprocessing.RandomFlip("horizontal",
                                                           input_shape=(IMG_SIZE,
                                                                        IMG_SIZE,
                                                                        3)),
        keras.layers.experimental.preprocessing.RandomRotation(0.1),
        keras.layers.experimental.preprocessing.RandomZoom(0.1),
    ]
)

# load the resnet50 model
resnet50 = tf.keras.applications.ResNet101(
    include_top=False,
    weights="imagenet",
    input_shape=(IMG_SIZE, IMG_SIZE, 3),
    pooling='avg'
)

# freeze the layers of the model
resnet50.trainable = False

# build the model
model = keras.Sequential([
    data_augmentation,
    normalization_layer,
    resnet50,
    keras.layers.Dense(2, activation='softmax')
])

# compile the model
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# train the model
start = time.time()
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS
)
end = time.time()

# save time in text file in benchmark folder
with open("../benchmark/resnet-101-time.txt", "w") as f:
    f.write("Time Taken for Resnet 101: " + str(end - start))

# save history in a csv file
pd.DataFrame(history.history).to_csv("../benchmark/resnet-101-history.csv")


# plot the accuracy and loss
plt.plot(history.history['accuracy'], label='accuracy')
plt.plot(history.history['val_accuracy'], label='val_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.ylim([0.5, 1])
plt.legend(loc='lower right')
plt.savefig('../benchmark/resnet-101-accuracy.png')
plt.show()

test_loss, test_acc = model.evaluate(val_ds, verbose=2)

print(test_acc)

