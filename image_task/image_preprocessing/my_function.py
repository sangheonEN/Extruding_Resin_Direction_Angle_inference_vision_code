#%%
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

ax.set_xlim([0, 3])
ax.set_ylim([0, 3])

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
import math

x = np.arange(0.000001, 1., 0.01, dtype=float)
p1 = 0.35
p2 = 0.88
sig1 = 0.0
sig2 = 1.0
# w1 = (1+(1/(math.exp(1)**sig1))) * (1 - (sig1 / (math.exp(1)+1)))
# w2 =(1+(1/(math.exp(1)**sig2))) * (1 - (sig2 / (math.exp(1)+1)))
w1 = 1/np.exp(sig1) + 1
w2 = 1/np.exp(sig2) + 1

print(w1, w2)

# y = [(-0.25*2*(1-num)**2*math.log(num)) for num in x]

# y2 = [(-0.25*(1-num)**2*math.log(num)) for num in x]
# y3 = [(-0.25*w1*(1-num)**2*math.log(num)) for num in x]
# y4 = [(-0.25*w2*(1-num)**2*math.log(num)) for num in x]
# fl_dot = -0.25*(1-p2)**2*math.log(p2)
# fl_dot2 = -0.25*(1-p1)**2*math.log(p1)
# cfl_dot = -0.25*w1*(1-p2)**2*math.log(p2)
# cfl_dot2 = -0.25*w2*(1-p1)**2*math.log(p1)
# cfl_dot3 = -0.25*w1*(1-p1)**2*math.log(p1)
# cfl_dot4 = -0.25*w2*(1-p2)**2*math.log(p2)

focal = [(-(1-num)**2*math.log(num)) for num in x]
erfl_sig1 = [(-w1*(1-num)**2*math.log(num)) for num in x]
erfl_sig2 = [(-w2*(1-num)**2*math.log(num)) for num in x]
ce = [-1*math.log(num) for num in x]
fl_dot = -(1-p2)**2*math.log(p2)
fl_dot2 = -(1-p1)**2*math.log(p1)
erfl_dot = -w1*(1-p2)**2*math.log(p2)
erfl_dot2 = -w2*(1-p1)**2*math.log(p1)
erfl_dot3 = -w1*(1-p1)**2*math.log(p1)
erfl_dot4 = -w2*(1-p2)**2*math.log(p2)
ce_dot = -1*math.log(p1)

# y3 = [(-0.25*1.37*(1-num)**2*math.log(num)) for num in x]

# y0 = np.arange(-1., 5., 0.01, dtype=float)
# x0 = np.zeros((y0.shape))
# x1 = np.ones((y0.shape))
# x2 = 1
# y2 = 1/np.exp(x2) + 1

# x3 = 0
# y3 = 1/np.exp(x3) + 1
#
# x4 = np.arange(-3., 0, 0.01, dtype=float)
# y4 = np.ones((x4.shape)) * 2
# x5 = np.arange(-3., 1., 0.01, dtype=float)
# y5 = np.ones((x5.shape)) * 1.37


fig, ax = plt.subplots(figsize=(5, 5))

# b = np.arange(0.0, 1.0, 0.2)
#
# a = list()
# for i in b:
#     a.append(round(i, 2))

ax.set_xlim([0.0, 1.0])
# ax.set_xticks(a)
ax.set_ylim([0.0, 5])

# ax.plot(x, y, c='green', label= 'CFL')
ax.plot(x, erfl_sig1, c='purple', linestyle='dashed')
ax.plot(x, erfl_sig2, c='green', linestyle='dashed')
ax.fill_between(x, erfl_sig1, erfl_sig2, color='yellow', alpha=0.5)
ax.plot(x, focal, c='red')
# ax.plot(x, ce, c='gray')
ax.scatter(p1, fl_dot2, c="red")
ax.scatter(p1, erfl_dot2, c="green")
ax.scatter(p1, erfl_dot3, c="purple")
# ax.scatter(p1, ce_dot, c="gray")
ax.scatter(p2, fl_dot)
ax.scatter(p2, erfl_dot)
ax.scatter(p2, erfl_dot4)
ax.text(p1+0.05, fl_dot2, s= str((p1, round(fl_dot2, 2))), c='red')
ax.text(p1+0.05, erfl_dot2+0.05, s= str((p1, round(erfl_dot2, 2))), c='green')
ax.text(p1+0.05, erfl_dot3, s= str((p1, round(erfl_dot3, 2))), c='purple')
# ax.text(p1+0.05, ce_dot, s= str((p1, round(ce_dot, 3))), c='gray')
# 범례
# ax.legend()
# 축제목
ax.set_xlabel("probability of ground truth class")
ax.set_ylabel("loss")

# ax.plot(x, y3, c='black')
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

