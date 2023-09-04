# import matplotlib.pyplot as plt
import tensorflow as tf

y_true = [[0., 1.], [0., 0.]]
y_pred = [[1., 1.], [1., 0.]]
mse = tf.keras.losses.MeanSquaredError()
print(mse(y_true, y_pred).numpy())

y_true = [[0., 1.], [0., 0.]]
y_pred = [[0.6, 0.4], [0.4, 0.6]]
bce = tf.keras.losses.BinaryCrossentropy()
print(bce(y_true, y_pred).numpy())

# from_logits=False (default)
y_true = [[0, 1, 0], [0, 0, 1]]	# 원-핫 표현으로 부류를 나타낸다. 
y_pred = [[0.05, 0.95, 0], [0.1, 0.8, 0.1]]
cce = tf.keras.losses.CategoricalCrossentropy()
print(cce(y_true, y_pred).numpy())
cce2 = tf.keras.losses.CategoricalCrossentropy(from_logits=True)
print(cce2(y_true, y_pred).numpy())

y_true = [1, 2]		# 정수로 부류를 나타낸다. 0, 1, 2가 가능함
y_pred = [[0.05, 0.95, 0], [0.1, 0.8, 0.1]]
scce = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False)
print(scce(y_true, y_pred).numpy())
scce2 = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
print(scce2(y_true, y_pred).numpy())

m = tf.keras.metrics.Accuracy()
m.update_state(	[[1], [2], [3], [4]], 
			[[0], [2], [3], [4]])
print(m.result().numpy())

m = tf.keras.metrics.CategoricalAccuracy()
m.update_state([[0,   0,     1], [0,    1,     0]], 
               [[0.1, 0.9, 0.8], [0.05, 0.95, 0]])
print(m.result().numpy())

