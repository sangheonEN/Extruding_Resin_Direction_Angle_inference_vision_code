import numpy as np
import cv2
import os
import matplotlib.pyplot as plt



def dist(coordi_1, coordi_2):

    print(coordi_1)
    print(coordi_2)
    print(f"dist: {np.linalg.norm(coordi_1 - coordi_2)}")


    return np.linalg.norm(coordi_1 - coordi_2)


def extract_three_points(sort_left_curve_array):

    start_idx = 0
    mid_idx = len(sort_left_curve_array) / 2
    end_idx = len(sort_left_curve_array) - 1

    start_point = sort_left_curve_array[start_idx]
    mid_point = sort_left_curve_array[int(mid_idx)]
    end_point = sort_left_curve_array[end_idx]

    return start_point, mid_point, end_point


if __name__ == "__main__":

    print(os.path.dirname(os.path.abspath(__file__)))

    mask_file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "test_data", 'Label_35.png')

    # img shape: h, w, c
    img = cv2.imread(mask_file_path, cv2.IMREAD_GRAYSCALE)

    img_label_index = np.unique(img)

    # left or right 좌표 없을때 continue
    # if len(img_label_index) < 2:
    #     continue

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
    start, mid, end = extract_three_points(left_curve_array[coordinate_order])


    print(f"start point: {start}\n")
    print(f"mid point: {mid}\n")
    print(f"end point: {end}\n")
    print(f"all points {left_curve_array[coordinate_order]}")