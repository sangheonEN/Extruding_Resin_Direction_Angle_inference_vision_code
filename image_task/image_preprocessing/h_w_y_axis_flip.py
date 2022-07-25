import numpy as np
import cv2
import os
import matplotlib.pyplot as plt



def dist(coordi_1, coordi_2):

    print(coordi_1)
    print(coordi_2)
    print(f"dist: {np.linalg.norm(coordi_1 - coordi_2)}")


    return np.linalg.norm(coordi_1 - coordi_2)




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

    # order save
    while True:

        

        dist(initial_coordinate, next_coordinate)


        if len(coordinate_order) == points_num:
            break









    print("z")


