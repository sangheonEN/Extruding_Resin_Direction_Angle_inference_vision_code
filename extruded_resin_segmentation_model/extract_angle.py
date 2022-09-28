import numpy as np
import os
import cv2
import matplotlib.pyplot as plt
import math

def dist(coordi_1, coordi_2):

    print(coordi_1)
    print(coordi_2)
    print(f"dist: {np.linalg.norm(coordi_1 - coordi_2)}")


    return np.linalg.norm(coordi_1 - coordi_2)


def start_mid_end_f(sort_left_curve_array):

    start_idx = 0
    mid_idx = len(sort_left_curve_array) / 2
    end_idx = len(sort_left_curve_array) - 1

    start_point = sort_left_curve_array[start_idx]
    mid_point = sort_left_curve_array[int(mid_idx)]
    end_point = sort_left_curve_array[end_idx]

    return start_point, mid_point, end_point


def extract_three_points(img):

    # print(os.path.dirname(os.path.abspath(__file__)))
    #
    # mask_file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "test_data", 'Label_35.png')
    # img = cv2.imread(mask_file_path, cv2.IMREAD_GRAYSCALE)

    # img shape: h, w, c
    img_label_index = np.unique(img)

    # left, right curve 좌표 추출. np.where -> return is tuple. 1. idx [0] is height. 2. idx [1] is width.
    left_curve = np.where(img == img_label_index[2])
    right_curve = np.where(img == img_label_index[3])

    left_curve_h = left_curve[0]
    left_curve_w = left_curve[1]

    right_curve_h = right_curve[0]
    right_curve_w = right_curve[1]

    # x = 0 으로 평행이동
    left_curve_height_max = max(left_curve_h)
    left_curve_h -= left_curve_height_max
    # y 축 기준으로 flip
    left_curve_h *= -1

    # x = 0 으로 평행이동
    right_curve_height_max = max(right_curve_h)
    right_curve_h -= right_curve_height_max
    # y 축 기준으로 flip
    right_curve_h *= -1

    plt.scatter(left_curve_w, left_curve_h, c='red', s=1)
    plt.scatter(right_curve_w, right_curve_h, c='blue', s=1)
    plt.xlim(min(min(left_curve_w), min(right_curve_w))-10, max(max(left_curve_w), max(right_curve_w))+10)
    plt.ylim(min(min(left_curve_h), min(right_curve_h))-10, max(max(left_curve_h), max(right_curve_h))+10)
    # plt.xticks()
    # plt.yticks()
    plt.show()

    curve_coordinates = dict(left_curve=[left_curve_w, left_curve_h], right_curve=[right_curve_w, right_curve_h])

    # 점 가지고 첫 점, 끝 점 중앙점 찾기 코드 distance로!!
    coordinate_order = list()
    points_num = len(left_curve_h)

    start_idx = np.where(left_curve_h == min(left_curve_h))
    start_idx = start_idx[0][0]
    coordinate_order.append(start_idx)

    # n, (w, h)
    left_curve_array = np.array([left_curve_w, left_curve_h]).transpose(1, 0)
    initial_coordinate = left_curve_array[start_idx]

    # 시작하는 좌표 점 계산 flag, 그 후에는 new idx coordinate로 전환
    initial_flag = True
    # order save
    while True:

        if len(coordinate_order) == points_num:
            break

        # 거리 계산 저장할 array
        distance_array = np.zeros((len(left_curve_array)))

        # 외곽선 점 좌표끼리 거리계산
        for idx, next_coordinate in enumerate(left_curve_array):
            # 재방문 안되게 하기.
            if idx in coordinate_order:
                continue
            # 거리계산
            if initial_flag == True:
                distance = dist(initial_coordinate, next_coordinate)
            else:
                distance = dist(left_curve_array[next_coordinate_idx], next_coordinate)
            # 거리 array에 index별로 distance 저장
            distance_array[idx] = distance

        # 대신 distance 계산이 안되거나 본인 index면 0으로 저장되니 index는 inf화 하여 min 값 계산 안되도록 치환
        distance_array[distance_array <= 0] = float("inf")
        # 가장 작은 distance를 가지는 index 뽑아서 다음 순서로 저장.
        new_idx = np.where(distance_array == min(distance_array))
        if len(new_idx[0]) >= 2:
            for i in new_idx[0]:
                coordinate_order.append(i)
                next_coordinate_idx = i
        else:
            coordinate_order.append(new_idx[0][0])
            next_coordinate_idx = new_idx[0][0]
        initial_flag = False


    # extract three points
    start, mid, end = start_mid_end_f(left_curve_array[coordinate_order])

    # print(f"start point: {start}\n")
    # print(f"mid point: {mid}\n")
    # print(f"end point: {end}\n")
    # print(f"all points {left_curve_array[coordinate_order]}")

    return start, mid, end


def find_circle_and_extract_angle_using_three_points(x, y):
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
    d1 = (x[1] - x[0]) / (y[1] - y[0]);
    d2 = (x[2] - x[1]) / (y[2] - y[1]);

    cx = ((y[2] - y[0]) + (x[1] + x[2]) * d2 - (x[0] + x[1]) * d1) / (2 * (d2 - d1))
    cy = -d1 * (cx - (x[0] + x[1]) / 2) + (y[0] + y[1]) / 2

    r = math.sqrt((x[0] - cx) ** 2 + (y[0] - cy) ** 2)

    print(cx)
    print(cy)
    print(r)

    # x.append(cx)
    # y.append(cy)

    # 접선방정식 구하기 m = 기울기, n = y 절편
    m = (-1) * ((cx - x[2]) / (cy - y[2]))
    n = (-1) * (m * x[2]) + y[2]

    min_x = 0.8 * min(x)
    max_x = 1.2 * max(x)

    x_range = np.arange(min_x, max_x, 0.5)
    y_range = [(m * num + n) for num in x_range]

    print(f"y = {m}x + {n}")

    angle_rad = math.atan2((y_range[-1] - y_range[0]), (x_range[-1] - x_range[0]))
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

    return angle_deg


if __name__ == '__main__':

    pass
