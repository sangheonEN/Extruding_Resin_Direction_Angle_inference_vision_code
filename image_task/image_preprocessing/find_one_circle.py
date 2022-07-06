import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import tensorflow as tf
import math
import time
import cv2
import os

# tf.enable_eager_execution()


def gauss_newton():
    """_summary_
    gauss newton 방법을 이용한 비선형 최소자승법으로 2차평면의 점좌표(x, y)를 가지고 원을 근사하여 파라미터를 추정(cx, cy, r)
    
    
    """
    
    pass


def using_three_points(x, y): 
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
    
    # 임의의 데이터 생성
    # x = [2, 5, 10]
    # y = [5, 4, 2]
    
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
    
    # x.append(cx)
    # y.append(cy)
    
    
    # 접선방정식 구하기 m = 기울기, n = y 절편
    m = (-1)*((cx-x[2])/(cy-y[2]))
    n = (-1)*(m*x[2]) + y[2]

    min_x = 0.8 * min(x)
    max_x = 1.2 * max(x)


    x_range = np.arange(min_x,max_x, 0.5)
    y_range = [(m*num+n) for num in x_range]
  
    print(f"y = {m}x + {n}")
    
    angle_rad = math.atan2((y_range[-1]-y_range[0]), (x_range[-1]-x_range[0]))
    angle_deg = np.rad2deg(angle_rad)
    
    print(f"angle: {angle_deg}")  
    
    circle1 = plt.Circle((cx, cy), r, color='r', fill=False)
    
    fig, ax = plt.subplots(figsize=(10, 10))
    
    ax.set_xlim([500, 900])
    ax.set_ylim([500, 900])

    
    ax.scatter(x, y)
    ax.add_patch(circle1)
    ax.plot([cx, x[2]], [cy, y[2]])
    ax.plot(x_range, y_range)
    
    fig.savefig("./aa.png")
       

def gradient_descent(x, y):
    
    # 임의의 데이터 생성
    # data load x는 random sample, y는 원방정식을 기준으로 bais를 주어 생성

    # x = [np.random.rand() * 10 for i in range(100)]
    # x = np.array(x)
    # y = list()
    
    # for i in range(len(x)):
    #     y_item = np.sqrt(25 - np.square(x[i]-5))
    #     if np.random.rand() < 0.5:
    #         y_item = -y_item + 5.0
    #     else:
    #         y_item = y_item + 5.0
    #     y.append(y_item)

    # y = np.array(y) + np.random.rand(100) * 4 - 2
    
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

def extract_three_points(left_w, left_h):
    mid_idx = int(len(left_h) / 2)
    start_idx = np.argmax(left_h)
    end_idx = np.argmin(left_h)

    start_point_w, start_point_h = left_w[start_idx], left_h[start_idx]
    mid_point_w, mid_point_h = left_w[mid_idx], left_h[mid_idx]
    end_point_w, end_point_h = left_w[end_idx], left_h[end_idx]

    x_points_list = list()
    y_points_list = list()

    x_points_list.append(start_point_w)
    x_points_list.append(mid_point_w)
    x_points_list.append(end_point_w)
    y_points_list.append(start_point_h)
    y_points_list.append(mid_point_h)
    y_points_list.append(end_point_h)

    return x_points_list, y_points_list



if __name__ == "__main__":

    img_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'test_data', 'Label_29.png')

    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

    print(np.unique(img))

    # pixel value 1 is object and 2 is curve
    # img[img<=1.] = 0

    object = img == 1.

    curve = img>=2.
    curve = curve * 255.

    curve_h, curve_w = np.where(curve == 255.)

    mean = np.mean(curve_w)

    left_curve_idx = np.argwhere(curve_w <= mean)
    right_curve_idx = np.argwhere(curve_w >= mean)

    left_w = [curve_w[idx] for idx in left_curve_idx]
    left_h = [curve_h[idx] for idx in left_curve_idx]

    right_w = [curve_w[idx] for idx in right_curve_idx]
    right_h = [curve_h[idx] for idx in right_curve_idx]

    left_x_points, left_y_points = extract_three_points(left_w, left_h)
    right_x_points, right_y_points = extract_three_points(right_w, right_h)
        
    # gradient_descent
    # gradient_descent()
    
    # using three point 
    print(f"points info: x:{left_x_points}, y: {left_y_points}")
    using_three_points(left_x_points, left_y_points)
    

