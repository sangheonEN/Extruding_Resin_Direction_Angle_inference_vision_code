import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import tensorflow as tf
import math
import scipy
import time

tf.enable_eager_execution()


def gauss_newton():
    """_summary_
    gauss newton 방법을 이용한 비선형 최소자승법으로 2차평면의 점좌표(x, y)를 가지고 원을 근사하여 파라미터를 추정(cx, cy, r)
    
    
    """
    
    pass


def using_three_points(): 
    """
    곡선의 시작, 중점, 끝점을 추출하여 세 점을 이용해서 원 파라미터를 구함.  http://heilow.egloos.com/v/418569
    input (x1, y1), (x2, y2), (x3, y3)
    output cx, cy, r
    _summary_
        @brief 세점을 지나는 원의 중심점과 반지름을 구한다.
        @param * r:  원의 반지름
        @param * psRelMatchPos:  원의 중심점 좌표
        @param * psCenterPoint:  세점들의 좌표
        @return Error code
        
    1. figure는 전체 하얀 종이, Axes는 네모로 나누어지는 구간이라고 생각하시면 됩니다. plt.subplots()에 입력하는 ncols, nrows에 따라 가로 세로 공간이 나누어집니다.
    2. 네 그렇습니다. 참고로 nrows=2, ncols=2로 입력하시면 네 개로 나누어집니다. 이 때는 2차원 배열이 되기 때문에 ax[1,0] 식으로 지정해야 합니다.
    3. 객체를 Axes에 붙이는 역할을 합니다. 포스트잇을 A4에 붙이듯 딱 붙이는 명령입니다.
        
    """
    
    x = [2, 5, 10]
    y = [5, 4, 2]
    
    # x = [2, 10, 6]
    # y = [5, 3, 2]
    
    # circle 구하기 http://egloos.zum.com/heilow/v/418569
    d1=(x[1]-x[0])/(y[1]-y[0]);
    d2=(x[2]-x[1])/(y[2]-y[1]);
    
    cx=((y[2]-y[0])+(x[1]+x[2])*d2-(x[0]+x[1])*d1)/(2*(d2-d1))
    cy=-d1*(cx-(x[0]+x[1])/2)+(y[0]+y[1])/2
    
    r = math.sqrt((x[0]-cx)**2+(y[0]-cy)**2)
    
    print(cx)
    print(cy)
    print(r)
    
    x.append(cx)
    y.append(cy)
    
    
    # 접선방정식 구하기 m = 기울기, n = y 절편
    m = (-1)*((cx-x[2])/[cy-y[2]])
    n = (-1)*(m*x[2]) + y[2]
    
    x_line_point_list = list()
    y_line_point_list = list()
    
    x_val1 = 1
    x_val2 = 2
    
    y_1 = m*x_val1 + n
    y_2 = m*x_val2 + n
    
    x_line_point_list.append(x_val1)
    x_line_point_list.append(x_val2)    
    y_line_point_list.append(y_1)    
    y_line_point_list.append(y_2)    
    
    
    
    
    circle1 = plt.Circle((cx, cy), r, color='r', fill=False)
    
    fig, ax = plt.subplots(figsize=(10, 10))
    
    ax.set_xlim([-100, 100])
    ax.set_ylim([-100, 100])

    
    ax.scatter(x, y)
    ax.add_patch(circle1)
    ax.plot([cx, x[2]], [cy, y[2]])
    
    fig.savefig("./aa.png")
       

def gradient_descent():
    
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

    # iteration
    t = time.time()
    for e in range(4000):
        index = np.random.randint(0, 100, 10)
        with tf.GradientTape() as tape:
            dist = tf.square(x_center - x[index]) + tf.square(y_center - y[index])
            loss = tf.keras.losses.mse(dist, r_square)

        # gradient descent
        grad = tape.gradient(loss, [x_center, y_center, r_square])
        x_center.assign(x_center - lr*grad[0])
        y_center.assign(y_center - lr*grad[1])
        r_square.assign(r_square - lr*grad[2])

        x_c_list.append(x_center.numpy())
        y_c_list.append(y_center.numpy())
        r_square_list.append(r_square.numpy())

    t2 = time.time()
    time_result = time.localtime(t2 - t)
    print(f"time: {time_result.tm_sec}")

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


if __name__ == "__main__":
    
    
    # data load x는 random sample, y는 원방정식을 기준으로 bais를 주어 생성
    
  
    
    # gradient_descent
    gradient_descent()
    
    # using three point 
    # using_three_points()
    

