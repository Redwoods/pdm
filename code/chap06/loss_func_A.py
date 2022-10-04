import numpy as np
import matplotlib.pylab as plt

def MSE(target, y):
	return 0.5 * np.sum((y-target)**2)

y = np.array([ 0.0, 0.0, 0.8, 0.1, 0.0, 0.0, 0.0, 0.1, 0.0, 0.0 ])
target = np.array([ 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 ])
plt.scatter(y, target, c=target, s=100)
plt.show()
print(MSE(target, y))

target = np.array([ 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 ])
y = np.array([ 0.9, 0.0, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 ] )
plt.scatter(y, target, c=target, s=100)
plt.show()
print(MSE(target, y))

# 오차 0 %


# 오차 100 %