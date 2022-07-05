import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
# import tensorflow as tf
import math
import scipy

# tf.enable_eager_execution()


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
    """
    
    # sympy나 scipy
    
    x = [2, 5, 10]
    y = [5, 4, 2]
    
    d1=(x[1]-x[0])/(y[1]-y[0]);
    d2=(x[2]-x[1])/(y[2]-y[1]);
    
    cx=((y[2]-y[0])+(x[1]+x[2])*d2-(x[0]+x[1])*d1)/(2*(d2-d1))
    cy=-d1*(cx-(x[0]+x[1])/2)+(y[0]+y[1])/2
    
    r = math.sqrt((x[0]-cx)**2+(y[0]-cy)**2)
    
    print(cx)
    print(cy)
    print(r)
    
    circle1 = plt.Circle((0, 0), 0.2, color='r', fill=False)
    # scatt = plt.scatter(x, y)
    
    fig, ax = plt.subplots(nrows=2, ncols=1, figsize=(10, 10))
    
    ax[0].set_xlim([-100, 100])
    ax[0].set_ylim([-100, 100])
    ax[1].set_xlim([-100, 100])
    ax[1].set_ylim([-100, 100])
    
    ax[0].scatter(x, y)
    ax[1].add_patch(circle1)
    
    fig.savefig("./aa.png")
       
    # ATF_ERR_T CiADAS_ReconstructorDlg ::Get_Three_Point_Circle(float* r, AT_FPOINT* psRelMatchPos, const AT_FPOINT* psCenterPoint)
    # {
    #     float d1 = (psCenterPoint[1].x - psCenterPoint[0].x)/(psCenterPoint[1].y - psCenterPoint[0].y); 
    #     float d2 = (psCenterPoint[2].x - psCenterPoint[1].x)/(psCenterPoint[2].y - psCenterPoint[1].y); 
        
    #     float cx = ((psCenterPoint[2].y - psCenterPoint[0].y) + (psCenterPoint[1].x + psCenterPoint[2].x) * d2 - (psCenterPoint[0].x + psCenterPoint[1].x) * d1)/(2 * (d2 - d1)); 
    #     float cy = (-d1 * (cx-(psCenterPoint[0].x + psCenterPoint[1].x)/2) + (psCenterPoint[0].y + psCenterPoint[1].y)/2 );
        
    #     psRelMatchPos->x = cx;
    #     psRelMatchPos->y = cy;
    #     *r = sqrt(pow((float)(psCenterPoint[0].x - cx), 2) + pow((float)(psCenterPoint[0].y - cy), 2));
    #     return ATF_ERR_NONE;
    # }
        


def gradient_descent():
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
    for e in range(10000):
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
    
    
    # gradient_descent
    # gradient_descent()
    
    # using three point 
    using_three_points()
    

