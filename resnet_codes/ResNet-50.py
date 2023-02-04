import os
import warnings

import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow import keras
sns.set()


warnings.filterwarnings("ignore")

classes = ['COVID-19', 'NON-COVID-19']

# Path to the data directory
non_enhanced = os.listdir("../data/non-enhanced")


# parameters for resnet50
IMG_SIZE = 224
BATCH_SIZE = 32
EPOCHS = 10

# create a data generator
train_ds = tf.keras.preprocessing.image_dataset_from_directory(
    '../data/non-enhanced',
    validation_split=0.2,
    subset="training",
    seed=123,
    image_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE)

val_ds = tf.keras.preprocessing.image_dataset_from_directory(
    '../data/non-enhanced',
    validation_split=0.2,
    subset="validation",
    seed=123,
    image_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE)

class_names = train_ds.class_names
print(class_names)

# visualize the data
plt.figure(figsize=(10, 10))
for images, labels in train_ds.take(1):
    for i in range(9):
        ax = plt.subplot(3, 3, i + 1)
        plt.imshow(images[i].numpy().astype("uint8"))
        plt.title(class_names[labels[i]])
        plt.axis("off")
    plt.show()


# this part modifies 'train_ds' and 'val_ds' to improve performance during training
# train_ds is shuffled but val_ds is not
AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

# standardize the data
normalization_layer = tf.keras.layers.experimental.preprocessing.Rescaling(1./255)

# augment the data
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

# load resnet50
"""
include_top: whether to include the 3 fully-connected layers at the top of the network.

weights: one of None (random initialization) or 'imagenet' (pre-training on ImageNet).

input_shape: optional shape tuple, only to be specified if include_top is False (otherwise the input shape has to be 
(224, 224, 3) (with 'channels_last' data format) or (3, 224, 224) (with 'channels_first' data format). 
It should have exactly 3 inputs channels, and width and height should be no smaller than 197. E.g. (200, 200, 3) 
would be one valid value.

pooling: Optional pooling mode for feature extraction when include_top is False.
"""
resnet50 = tf.keras.applications.ResNet50(
    include_top=False,
    weights="imagenet",
    input_shape=(IMG_SIZE, IMG_SIZE, 3),
    pooling='avg'
)

# freeze the layers
resnet50.trainable = False

# create the model
model = keras.Sequential([
    data_augmentation,
    normalization_layer,
    resnet50,
    keras.layers.Dense(2, activation='softmax')
])

# compile the model
model.compile(
    optimizer='adam',
    loss=tf.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=['accuracy', tf.keras.metrics.Recall(), tf.keras.metrics.Precision(), tf.keras.metrics.AUC()]
)

# train the model
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS
)

# plot the results
plt.plot(history.history['accuracy'], label='accuracy')
plt.plot(history.history['val_accuracy'], label='val_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.ylim([0.5, 1])
plt.legend(loc='lower right')

test_loss, test_acc, recall, precision, f1_score = model.evaluate(val_ds, verbose=2)

print(test_acc, recall, precision, f1_score)
