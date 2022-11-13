import matplotlib.pyplot as plt 
from matplotlib.image import imread
import matplotlib.image as mpimg
import random

import tensorflow as tf
from tensorflow.keras import layers

# Data
import os
os.getcwd()

train_dir = './Petimages/train'
test_dir = './Petimages/test'

# sample image
image = imread('./PetImages/train/dog/1.jpg')
image.shape
plt.imshow(image)
plt.show()

# View a random image
def view_random_image(target_dir, target_class):
  # Setup target directory (we'll view images from here)
  target_folder = target_dir+'/'+target_class

  # Get a random image path
  random_image = random.sample(os.listdir(target_folder), 1)

  # Read in the image and plot it using matplotlib
  img = mpimg.imread(target_folder + "/" + random_image[0])
  plt.imshow(img, cmap = "gray")
  plt.title(target_class)
  plt.axis("off");

  print(f"Image shape: {img.shape}") # show the shape of the image

  return img

img = view_random_image(target_dir = train_dir, target_class = "cat")
img = tf.constant(img)
plt.show()

# Setting up the data
# 
# from tensorflow.keras.preprocessing.image import ImageDataGenerator

# train_datagen = ImageDataGenerator(rescale = 1./255, shear_range = 0.2,
#   zoom_range = 0.2, horizontal_flip = True)

# test_datagen = ImageDataGenerator(rescale = 1./255)

# train_generator = train_datagen.flow_from_directory(
#     train_dir,                      
#     target_size=(128, 128), # (180,180)
#     batch_size=20,          # 32,...
#     class_mode = 'binary')

# test_generator = test_datagen.flow_from_directory(
#     test_dir,
#     target_size=(128, 128),
#     batch_size=20,
#     class_mode = 'binary')

# import tensorflow as tf
from tensorflow.keras.preprocessing import image_dataset_from_directory
IMG_SIZE = (128, 128)
batchSize=20

print("Training Images:")
train_data = image_dataset_from_directory(directory = train_dir,
                                            image_size = IMG_SIZE,
                                            label_mode = "binary",
                                            color_mode = "rgb",
                                            batch_size = batchSize)

print("Testing Images:")
test_data = image_dataset_from_directory(directory = test_dir,
                                            image_size = IMG_SIZE,
                                            label_mode = "binary",
                                            color_mode = "rgb",
                                            batch_size = batchSize)

# Inspecting the train_data
train_data   # BatchDataset
# <BatchDataset element_spec=(TensorSpec(shape=(None, 128, 128, 3), dtype=tf.float32, name=None), 
# TensorSpec(shape=(None, 1), dtype=tf.float32, name=None))>

# Plot images from dataset
# figure 크기를 조절합니다.
plt.figure(figsize=(10, 10))
names = ['cat','dog']
# 배치 하나를 가져옵니다.
for images, labels in train_data.take(1):
    for i in range(9):
        ax = plt.subplot(3, 3, i + 1)
        plt.imshow(images[i].numpy().astype("uint8"))
        plt.title(names[int(labels[i])])
        plt.axis("off")
plt.show()

#
# Data augmentation
#
# Create a data augmentation stage with horizontal flipping, rotations, zooms
#
tf.get_logger().setLevel('ERROR')  # Clear warnings in data augmentation

from tensorflow import keras
data_augmentation = keras.Sequential([
  layers.RandomFlip("horizontal"),
  layers.RandomRotation(0.2),
  layers.RandomZoom(0.2),
  layers.RandomHeight(0.2),
  layers.RandomWidth(0.2),
  # layers.Rescaling(1./255) # keep for ResNet50V2, remove for EfficientNetB0
], name ="data_augmentation")


# Plot the augmented images
plt.figure(figsize=(10,10))
image_idx = 0
for images, labels in train_data.take(1):    # Make a batch of images & labels
    print(labels,images.shape)
    for i in range(9):
        ax = plt.subplot(3, 3, i + 1)
        aug_img = data_augmentation(tf.expand_dims(images[image_idx], axis=0))
        print(aug_img.shape)
        plt.imshow(aug_img[0].numpy().astype("uint8"))
        plt.title("{}".format(names[int(labels[image_idx])]))
        plt.axis("off")
    break
plt.show()

plt.figure(figsize=(10,10))
image_idx = 0
for images, labels in test_data.take(1):    # Make a batch of images & labels
    print(labels,images.shape)
    for i in range(9):
        ax = plt.subplot(3, 3, i + 1)
        aug_img = data_augmentation(tf.expand_dims(images[image_idx], axis=0))
        print(aug_img.shape)
        plt.imshow(aug_img[0].numpy().astype("uint8"))
        plt.title("{}".format(names[int(labels[image_idx])]))
        plt.axis("off")
    break
plt.show()

#
# Visualize data from data generator
# 1. Extract one batch
# for x_data, t_data in train_generator:
#     print(x_data.shape)  # (20, 128, 128, 3)
#     print(type(x_data))  # <class 'numpy.ndarray'>
#     print(t_data)        # [0. 1. 1. 1. 1. 0. 0. 1. 0. 0. 1. 0. 0. 1. 1. 0. 0. 0. 1. 0.]
#     # 0 : 고양이,  1 : 댕댕이
#     # break

# # 2. Display images in the batch
# fig = plt.figure(figsize=(15, 12))
# # axs = []
# for x_data, t_data in train_generator:
#     for idx, img in enumerate(x_data):
#         ax = plt.subplot(4, 5, idx + 1)
#         # axs.append(fig.add_subplot(4,5,idx+1))
#         plt.imshow(img)
#         plt.title("{}".format(str(int(t_data[idx]))))
#         plt.axis("off")
#     break
# plt.show()

# ## Creating the model 
# # Setting up the model
# Model - Sequential model
# model = models.Sequential()
# model.add(layers.Conv2D(32,(3,3), activation='relu', input_shape=(128,128,3)))
# model.add(layers.MaxPooling2D(2,2))
# model.add(layers.Conv2D(64,(3,3), activation='relu'))
# model.add(layers.MaxPooling2D(2,2))
# model.add(layers.Flatten())
# model.add(layers.Dense(units=512, activation='relu'))
# model.add(layers.Dense(units=1, activation='sigmoid'))
# 
# # Functional Model
# inputs = tf.keras.layers.Input(shape = (128,128,3), name = "Input_Layer")
# #x = tf.keras.layers.experimental.preprocessing.Rescaling(1./255)(inputs)
# x = data_augmentation(inputs)
# x = tf.keras.layers.Conv2D(32, (3,3), activation='relu', padding='same')(x)
# x = tf.keras.layers.MaxPooling2D(2,2)(x)
# x = tf.keras.layers.Conv2D(64,(3,3), activation='relu', padding='same')(x)
# x = tf.keras.layers.MaxPooling2D(2,2)(x)
# x = tf.keras.layers.Flatten()(x)
# x = layers.Dense(units=512, activation='relu')(x)
# outputs = tf.keras.layers.Dense(1, activation = "sigmoid")(x)
# model = tf.keras.Model(inputs, outputs)

# Sequential model with data augmentation
model = tf.keras.Sequential([
  layers.Input(shape=(128,128,3),name='input_layer'),
  data_augmentation,
  layers.Conv2D(32,3,activation='relu'),
  layers.MaxPool2D(pool_size=2),
  layers.Conv2D(64,3,activation='relu'),
  layers.MaxPool2D(pool_size=2),
  layers.Flatten(),
  layers.Dense(1,activation='sigmoid')
])

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001), 
              loss = 'binary_crossentropy', 
              metrics = ['accuracy'])

model.summary()
keras.utils.plot_model(model, show_shapes=True)

# Building the Model

# Setting up the callbacks
# Setup EarlyStopping callback to stop training if model's val_loss doesn't improve for 3 epochs
early_stopping = tf.keras.callbacks.EarlyStopping(monitor="val_loss", # watch the val loss metric
                                                  patience=10) # if val loss decreases for 10 epochs in a row, stop training
# Creating learning rate reduction callback
reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss",  
                                                 factor=0.25, # multiply the learning rate by 0.2 (reduce by 4x)
                                                 patience=5,
                                                 verbose=1, # print out when learning rate goes down 
                                                 min_lr=1e-7)

## Check the summary
for no, layer in enumerate(model.layers):
  print(no, layer.trainable)

#
######################################################
# Training model using augmentated data
######################################################
#
history = model.fit(train_data, 
                    epochs=100, 
                    steps_per_epoch = len(train_data), 
                    validation_data = test_data,
                    validation_steps = len(test_data), # batchSize,
                    callbacks = [early_stopping, reduce_lr])

#
model.evaluate(test_data)

# training graph
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.xlabel('Epoch')
plt.xlabel('Accuracy')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()

#############################################
# More training graphs
# More graphs of loss and accuracy
# import matplotlib.pyplot as plt
import numpy as np

history_dict = history.history 
loss = history_dict['loss']
val_loss = history_dict['val_loss']

epochs = range(1, len(loss) + 1)

plt.figure(figsize=(14, 4))

plt.subplot(1,2,1)
plt.plot(epochs, loss, 'go-', label='Training Loss')
plt.plot(epochs, val_loss, 'bd', label='Validation Loss')
plt.plot(np.argmin(np.array(val_loss))+1,val_loss[np.argmin(np.array(val_loss))], 'r*', ms=12)
plt.title('Training and Validation Loss, min: ' + str(np.round(val_loss[np.argmin(np.array(val_loss))],4)))
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

acc = history_dict['accuracy']
val_acc = history_dict['val_accuracy']

epochs = range(1, len(loss) + 1)

plt.subplot(1,2,2)
plt.plot(epochs, acc, 'go-', label='Training Accuracy') #, c='blue')
plt.plot(epochs, val_acc, 'bd', label='Validation Accuracy') #, c='red')
plt.plot(np.argmax(np.array(val_acc))+1,val_acc[np.argmax(np.array(val_acc))], 'r*', ms=12)
plt.title('Training and Validation Accuracy, max: ' + str(np.round(val_acc[np.argmax(np.array(val_acc))],4)))
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

