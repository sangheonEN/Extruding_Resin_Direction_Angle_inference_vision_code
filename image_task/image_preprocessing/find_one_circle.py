#%%
import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import tensorflow as tf

tf.enable_eager_execution()

# data load
x = [np.random.rand() * 10 for i in range(100)]
x = np.array(x)

y = list()
for i in range(len(x)):
    y_item = np.sqrt(25 - np.square(x[i]-5))
    if np.random.rand() < 0.5:
        y_item = -y_item + 5.0
    else:
        y_item = y_item + 5.0
    y.append(y_item)

y = np.array(y) + np.random.rand(100) * 4 - 2

# gradient descent

# parameters initialize
x_center = tf.Variable(1.0)
y_center = tf.Variable(1.0)
r_square = tf.Variable(1.0)
lr = 0.001

x_c_list = list()
y_c_list = list()
r_square_list = list()

for e in range(10000):
    index = np.random.randint(0, 100, 10)
    with tf.GradientTape() as tape:
        dist = tf.square(x_center - x[index]) + tf.square(y_center - y[index])
        loss = tf.keras.losses.mse(dist, r_square)

    grad = tape.gradient(loss, [x_center, y_center, r_square])

    x_center.assign(x_center - lr*grad[0])
    y_center.assign(y_center - lr*grad[1])
    r_square.assign(r_square - lr*grad[2])

    x_c_list.append(x_center.numpy())
    y_c_list.append(y_center.numpy())
    r_square_list.append(r_square.numpy())


print(x_center)
print(y_center)
print(r_square)

plt.subplot(5, 1, 1)
plt.scatter(x, y)
plt.subplot(5, 1, 2)
plt.Circle((x_center, y_center), r_square, fill=False)
plt.subplot(5, 1, 3)
plt.plot(x_c_list)
plt.subplot(5, 1, 4)
plt.plot(y_c_list)
plt.subplot(5, 1, 5)
plt.plot(r_square_list)

plt.show()




