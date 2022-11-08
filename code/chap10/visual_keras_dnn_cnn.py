#  https://www.analyticsvidhya.com/blog/2022/03/visualize-deep-learning-models-using-visualkeras/
#

import keras
import tensorflow as tf

from tensorflow import keras
from keras.models import Sequential
from tensorflow.keras.layers import Input, Conv2D, Dense, Flatten, Dropout
from tensorflow.keras.layers import GlobalMaxPooling2D, MaxPooling2D
from tensorflow.keras.models import Model
from tensorflow.keras import regularizers, optimizers

import visualkeras
from PIL import ImageFont
font = ImageFont.truetype("arial.ttf", 24)

# DNN
model = Sequential()
model.add(Dense(16, activation='relu', input_shape=(4000,)))
model.add(Dense(8, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['accuracy'])

visualkeras.layered_view(model, 
                         to_file='./figures/dnn.png', 
                         legend=True, font=font, spacing=50, 
                         scale_xy=10, one_dim_orientation='y')

visualkeras.layered_view(model, min_xy=10, min_z=10, # to_file='../figures/spam.png', 
                         scale_xy=100, scale_z=100, one_dim_orientation='x')

# visualkeras.graph_view(model)

# CNN-1
model = Sequential()
model.add(Conv2D(64,(4,4),input_shape=(32,32,3),activation='relu',padding='same'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Conv2D(32,(4,4),activation='relu',padding='same'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(10, activation='softmax'))

visualkeras.layered_view(model, 
                         legend=True, font=font, 
                         scale_xy=5, scale_z=1, 
                         spacing=60, one_dim_orientation='y')

visualkeras.layered_view(model, #min_xy=2, min_z=2
                         to_file='./figures/cnnn2fcn2.png', 
                         legend=True, font=font, spacing=50, 
                         scale_xy=5, scale_z=1, one_dim_orientation='x')


# CNN-2
model = Sequential()
model.add(Conv2D(64,(4,4),input_shape=(32,32,3),activation='relu',padding='same'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Conv2D(64,(4,4), activation='relu',padding='same'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.2))
model.add(Conv2D(128,(4,4), activation='relu',padding='same'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))
model.add(Conv2D(128,(4,4), activation='relu',padding='same'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.35))
model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(10, activation='softmax'))

visualkeras.layered_view(model, 
                         legend=True, font=font, spacing=20, 
                         one_dim_orientation='y')

visualkeras.layered_view(model, # min_xy=2, min_z=2, # to_file='../figures/spam.png', 
                         legend=True, font=font, spacing=25, 
                         scale_xy=10, scale_z=1, one_dim_orientation='y')


# customize the colours of the layers using the following code.

from tensorflow.keras import layers
from collections import defaultdict
color_map = defaultdict(dict)  # customize the colours
color_map[layers.Conv2D]['fill'] = '#00f5d4'
color_map[layers.MaxPooling2D]['fill'] = '#8338ec'
color_map[layers.Dropout]['fill'] = '#03045e'
color_map[layers.Dense]['fill'] = '#fb5607'
color_map[layers.Flatten]['fill'] = '#ffbe0b'
visualkeras.layered_view(model, legend=True, font=font,color_map=color_map)


# visualkeras.layered_view(model, to_file='./figures/dnn.png', 
#                          min_xy=10, min_z=10, 
#                          scale_xy=100, scale_z=100, 
#                          one_dim_orientation='x')

visualkeras.layered_view(model, to_file='./figures/dnn.png', 
                         legend=True, font=font)
                         # type_ignore=[ZeroPadding2D, Dropout, Flatten, visualkeras.SpacingDummyLayer])
