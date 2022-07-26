import os
import cv2
from h_w_y_axis_flip import *
from math import atan2


def pca_angle(img, img_path):

    # 이미지 좌표 y좌표 평행이동 및 y축 반전
    left_curve_w, left_curve_h, right_curve_w, right_curve_h = parallel_movement(img)

    # left, right 좌표 array 데이터
    left_curve_array = np.array([left_curve_w, left_curve_h], dtype=np.float64).transpose(1, 0)
    right_curve_array = np.array([right_curve_w, right_curve_h], dtype=np.float64).transpose(1, 0)

    # pca
    mean = np.empty((0))
    mean_left, eigenvectors_left, eigenvalues_left = cv2.PCACompute2(left_curve_array, mean)
    mean_right, eigenvectors_right, eigenvalues_right = cv2.PCACompute2(right_curve_array, mean)

    # 고유벡터와 x축 사이 angle 추출
    left_angle = atan2(eigenvectors_left[0, 1], eigenvectors_left[0, 0])
    right_angle = atan2(eigenvectors_right[0, 1], eigenvectors_right[0, 0])
    left_angle = int(np.rad2deg(left_angle))
    right_angle = int(np.rad2deg(right_angle))

    # 기울기가 음수인 직선 방정식이면, 1, 2분면으로 옮기고 180도에서 차감해서 각도를 맞춤.
    if left_angle < 0:
        left_angle = 180-(left_angle * (-1))
    else:
        pass

    if right_angle < 0:
        right_angle = 180-(right_angle * (-1))
    else:
        pass

    left_cntr = (int(mean_left[0, 0]), int(mean_left[0, 1]))
    right_cntr = (int(mean_right[0, 0]), int(mean_right[0, 1]))
    scale_factor = 30

    # circle_left = plt.Circle((left_cntr[0], left_cntr[1]), 1, color='r', fill=False)
    # circle_right = plt.Circle((right_cntr[0], right_cntr[1]), 1, color='r', fill=False)

    fig, ax = plt.subplots(figsize=(10, 10))

    ax.set_xlim([min(right_curve_array[:, 0] - 50), max(right_curve_array[:, 0]) + 50])
    ax.set_ylim([min(right_curve_array[:, 1] - 50), max(right_curve_array[:, 1] + 50)])

    ax.scatter(left_curve_array[:, 0], left_curve_array[:, 1])
    ax.scatter(right_curve_array[:, 0], right_curve_array[:, 1])
    # ax.add_patch(circle_left)
    # ax.add_patch(circle_right)
    # ax.plot([int(left_order_curve_array[0][0]), int(left_order_curve_array[0][0] + eigenvectors_left[0, 0] * scale_factor)], [int(left_order_curve_array[0][1]), int(left_order_curve_array[0][1] - eigenvectors_left[0, 1] * scale_factor)])
    # ax.plot([int(right_order_curve_array[0][0]), int(right_order_curve_array[0][0] + eigenvectors_right[0, 0] * scale_factor)], [int(right_order_curve_array[0][1]), int(right_order_curve_array[0][1] - eigenvectors_right[0, 1] * scale_factor)])

    ax.text(left_cntr[0]+30, left_cntr[1]+30, f"left_angle: {left_angle}", fontsize=10)
    ax.text(right_cntr[0]+30, right_cntr[1]+30, f"right_angle: {right_angle}", fontsize=10)

    if not os.path.exists("./pca_angle"):
        os.makedirs("./pca_angle")

    fig.savefig(f"./pca_angle/{img_path}")

    # # visualization
    # # Store the center of the object (x, y)
    # left_cntr = (int(mean_left[0, 0]), int(mean_left[0, 1]))
    # right_cntr = (int(mean_right[0, 0]), int(mean_right[0, 1]))
    #
    # scale_factor = 30
    #
    # # Draw the principal components
    # cv2.circle(img, left_cntr, 3, (255, 0, 255), 2)
    # # 이미지상 y축은 아래로 양수니까 위쪽으로 선을 긋기 위해 eigenvector[0, 1] y값은 원점에 -로 더해주어 스케일 업 해준다.
    # cv2.line(img, (int(left_cntr[0]), int(left_cntr[1])),
    #          (int(left_cntr[0] + eigenvectors_left[0, 0] * scale_factor), int(left_cntr[1] - eigenvectors_left[0, 1] * scale_factor)),
    #          (0, 0, 255), 1, cv2.LINE_AA)
    # cv2.putText(img, f"Rotation_ Angle {str(left_angle)}", (left_cntr[0] + 20, left_cntr[1] + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
    #             (0, 0, 255), 1, cv2.LINE_AA)
    #
    #
    # # Draw the principal components
    # cv2.circle(img, right_cntr, 3, (255, 0, 255), 2)
    # # 이미지상 y축은 아래로 양수니까 위쪽으로 선을 긋기 위해 eigenvector[0, 1] y값은 원점에 -로 더해주어 스케일 업 해준다.
    # cv2.line(img, (int(right_cntr[0]), int(right_cntr[1])),
    #          (int(right_cntr[0] + eigenvectors_right[0, 0] * scale_factor), int(right_cntr[1] - eigenvectors_right[0, 1] * scale_factor)),
    #          (255, 0, 0), 1, cv2.LINE_AA)
    # cv2.putText(img, f"Rotation_ Angle {str(right_angle)}", (right_cntr[0] + 20, right_cntr[1] + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
    #             (0, 0, 255), 1, cv2.LINE_AA)
    #
    # cv2.imwrite("./pca_angle.png", img)

    return left_angle, right_angle


def gradient_circle_angle(img):

    # 이미지 좌표 y좌표 평행이동 및 y축 반전
    left_curve_w, left_curve_h, right_curve_w, right_curve_h = parallel_movement(img)

    # 유클리디언 최소 거리 기준 좌표 정렬
    left_order_curve_array = order_curve_points(left_curve_w, left_curve_h)
    right_order_curve_array = order_curve_points(right_curve_w, right_curve_h)

    # gradient approximation circle


    # angle 추출


def three_points_circle_angle(img):

    # 이미지 좌표 y좌표 평행이동 및 y축 반전
    left_curve_w, left_curve_h, right_curve_w, right_curve_h = parallel_movement(img)

    # 유클리디언 최소 거리 기준 좌표 정렬
    left_order_curve_array = order_curve_points(left_curve_w, left_curve_h)
    right_order_curve_array = order_curve_points(right_curve_w, right_curve_h)

    # extract three points
    start_l, mid_l, end_l = extract_three_points(left_order_curve_array)
    start_r, mid_r, end_r = extract_three_points(right_order_curve_array)

    left_x_list = list()
    left_y_list = list()
    right_x_list = list()
    right_y_list = list()

    left_x_list.append(end_l[0])
    left_x_list.append(mid_l[0])
    left_x_list.append(start_l[0])
    left_y_list.append(end_l[1])
    left_y_list.append(mid_l[1])
    left_y_list.append(start_l[1])

    right_x_list.append(end_r[0])
    right_x_list.append(mid_r[0])
    right_x_list.append(start_r[0])
    right_y_list.append(end_r[1])
    right_y_list.append(mid_r[1])
    right_y_list.append(start_r[1])

    # angle 추출
    left_angle = find_one_circle.using_three_points(left_x_list, left_y_list, left_order_curve_array)
    right_angle = find_one_circle.using_three_points(right_x_list, right_y_list, right_order_curve_array)

    return left_angle, right_angle


if __name__ == "__main__":

    print(os.path.dirname(os.path.abspath(__file__)))

    mask_file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "test_data")

    image_list = os.listdir(mask_file_path)

    for image_path in image_list:

        # img shape: h, w, c
        img = cv2.imread(os.path.join(mask_file_path, image_path), cv2.IMREAD_GRAYSCALE)

        # left or right 좌표 없을때 continue
        img_label_index = np.unique(img)

        if len(img_label_index) < 4:
            continue

        # 세점을 이용한 circle 각도 추출
        # left_angle, right_angle = three_points_circle_angle(img)

        # pca를 이용한 외곽선 점 데이터 고유벡터 선형 각도 추출
        left_angle, right_angle = pca_angle(img, image_path)

        print(f"left_angle: {left_angle}\n")
        print(f"right_angle: {right_angle}\n")
