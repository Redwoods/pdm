import tensorflow as tf
from tensorflow.keras import layers


input_shape = (4, 28, 28, 3)
x = tf.random.normal(input_shape)
y = tf.keras.layers.Conv2D(2, 3, activation='relu', input_shape=input_shape[1:])(x)
print(y.shape)


input_shape = (4, 28, 28, 3)
x = tf.random.normal(input_shape)
y = layers.Conv2D(2, 3, activation='relu', padding="same", 
			input_shape=input_shape[1:])(x)
print(y.shape)


x = tf.constant([[1., 2., 3.],     [4., 5., 6.],     [7., 8., 9.]])
x = tf.reshape(x, [1, 3, 3, 1])
max_pool_2d = tf.keras.layers.MaxPooling2D(pool_size=(2, 2), 
                                           strides=(1, 1), padding='valid')
print(max_pool_2d(x))

# [DIY] test 'same' padding in MaxPooling2D layer
