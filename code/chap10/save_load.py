import numpy as np
import tensorflow as tf

test_input = np.random.random((128, 32))
test_target = np.random.random((128, 1))

inputs = tf.keras.Input(shape=(32,))
outputs = tf.keras.layers.Dense(1)(inputs)
model = tf.keras.Model(inputs, outputs)
model.compile(optimizer="adam", loss="mean_squared_error")
model.fit(test_input, test_target, epochs=3)

model.save("my_model")
# model.save('mymodel.hdf5')

saved_model = tf.keras.models.load_model("my_model")
saved_model.fit(test_input, test_target, epochs=3)
#
#  [DIY] save the new model as "mymodel2.hdf5"
#
