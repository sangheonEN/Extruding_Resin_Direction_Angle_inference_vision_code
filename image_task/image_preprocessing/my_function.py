import matplotlib.pyplot as plt
import numpy as np

x = np.arange(-1., 2., 0.01, dtype=float)
y = 1/np.exp(x) + 1

y0 = np.arange(-1., 5., 0.01, dtype=float)
x0 = np.zeros((y0.shape))
x1 = np.ones((y0.shape))
x2 = 1
y2 = 1/np.exp(x2) + 1

x3 = 0
y3 = 1/np.exp(x3) + 1

x4 = np.arange(-3., 0, 0.01, dtype=float)
y4 = np.ones((x4.shape)) * 2
x5 = np.arange(-3., 1., 0.01, dtype=float)
y5 = np.ones((x5.shape)) * 1.37


fig, ax = plt.subplots(figsize=(5, 5))

ax.set_xlim([-3, 3])
ax.set_ylim([-1, 5])

ax.plot(x, y, c='red')
ax.plot(x0, y0, c='blue', linewidth=1.0, linestyle = '--')
ax.plot(x1, y0, c='green', linewidth=1.0, linestyle = '--')
ax.plot(x4, y4, c='blue', linewidth=1.0, linestyle = '--')
ax.plot(x5, y5, c='green', linewidth=1.0, linestyle = '--')
ax.scatter(x2, y2, c='green', s=20)
ax.scatter(x3, y3, c='blue', s=20)
ax.text(x2+0.1, y2, s= str((x2, round(y2, 2))), c='green')
ax.text(x3+0.1, y3, s= str((x3, round(y3, 2))), c='blue')


fig.savefig("./my_function.png")

#%%
import matplotlib.pyplot as plt
import numpy as np

import matplotlib.pyplot as plt
import numpy as np
import math

x = np.arange(0.000001, 1., 0.01, dtype=float)

y = [(-0.25*2*(1-num)**2*math.log(num)) for num in x]
y2 = [(-0.25*1.37*(1-num)**2*math.log(num)) for num in x]

# y0 = np.arange(-1., 5., 0.01, dtype=float)
# x0 = np.zeros((y0.shape))
# x1 = np.ones((y0.shape))
# x2 = 1
# y2 = 1/np.exp(x2) + 1
#
# x3 = 0
# y3 = 1/np.exp(x3) + 1
#
# x4 = np.arange(-3., 0, 0.01, dtype=float)
# y4 = np.ones((x4.shape)) * 2
# x5 = np.arange(-3., 1., 0.01, dtype=float)
# y5 = np.ones((x5.shape)) * 1.37


fig, ax = plt.subplots(figsize=(5, 5))

ax.set_xlim([-3, 3])
ax.set_ylim([-1, 5])

ax.plot(x, y, c='red')
ax.plot(x, y2, c='blue')
# ax.plot(x0, y0, c='blue', linewidth=1.0, linestyle = '--')
# ax.plot(x1, y0, c='green', linewidth=1.0, linestyle = '--')
# ax.plot(x4, y4, c='blue', linewidth=1.0, linestyle = '--')
# ax.plot(x5, y5, c='green', linewidth=1.0, linestyle = '--')
# ax.scatter(x2, y2, c='green', s=20)
# ax.scatter(x3, y3, c='blue', s=20)
# ax.text(x2+0.1, y2, s= str((x2, round(y2, 2))), c='green')
# ax.text(x3+0.1, y3, s= str((x3, round(y3, 2))), c='blue')


fig.savefig("./focal.png")

#%%

