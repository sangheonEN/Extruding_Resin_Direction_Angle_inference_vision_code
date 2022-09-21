import os
import cv2
import numpy as np

from h_w_y_axis_flip import *
from math import atan2
from math import cos, sin, sqrt, pi


LEE_COLORMAP = [
    [0, 0, 0],
    [255, 255, 255],
    [255, 0, 0],
    [0, 0, 255]
]

def visualization(label_mask):
    label_colours = LEE_COLORMAP

    n_classes = np.unique(label_mask)

    r = label_mask.copy()
    g = label_mask.copy()
    b = label_mask.copy()
    for ll in range(0, len(n_classes)):
        r[label_mask == ll] = label_colours[ll][0]
        g[label_mask == ll] = label_colours[ll][1]
        b[label_mask == ll] = label_colours[ll][2]
    rgb = np.zeros((label_mask.shape[0], label_mask.shape[1], 3))
    rgb[:, :, 0] = b.squeeze()
    rgb[:, :, 1] = g.squeeze()
    rgb[:, :, 2] = r.squeeze()

    return rgb


def angle_(p_, q_):
    p = list(p_)
    q = list(q_)

    angle = atan2(p[1] - q[1], p[0] - q[0]) # angle in radians

    # hypotenuse = sqrt((p[1] - q[1]) * (p[1] - q[1]) + (p[0] - q[0]) * (p[0] - q[0]))
    # # Here we lengthen the arrow by a factor of scale
    # q[0] = p[0] - scale * hypotenuse * cos(angle)
    # q[1] = p[1] - scale * hypotenuse * sin(angle)
    # cv2.line(img, (mean_exflusion_h, mean_exflusion_w), (int(q[0]), int(q[1])), colour, 1, cv2.LINE_AA)
    # # create the arrow hooks
    # p[0] = q[0] + 9 * cos(angle + pi / 4)
    # p[1] = q[1] + 9 * sin(angle + pi / 4)
    # cv2.line(img, (mean_exflusion_h, mean_exflusion_w), (int(q[0]), int(q[1])), colour, 1, cv2.LINE_AA)
    # p[0] = q[0] + 9 * cos(angle - pi / 4)
    # p[1] = q[1] + 9 * sin(angle - pi / 4)
    # cv2.line(img, (mean_exflusion_h, mean_exflusion_w), (int(q[0]), int(q[1])), colour, 1, cv2.LINE_AA)
    return angle

def front_pca_angle_case0234(img, img_path):
    # 이미지 좌표 y좌표 평행이동 및 y축 반전

    img_label_index = np.unique(img)

    rgb = visualization(img)

    exflusion = np.where(img == img_label_index[1])
    left_curve = np.where(img == img_label_index[2])
    right_curve = np.where(img == img_label_index[3])

    left_curve_h = left_curve[0]
    # y 축 기준으로 flip
    left_curve_h *= -1
    left_curve_w = left_curve[1]

    right_curve_h = right_curve[0]
    # y 축 기준으로 flip
    right_curve_h *= -1
    right_curve_w = right_curve[1]

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


    left_cntr = (int(mean_left[0, 0]), int(mean_left[0, 1]))
    right_cntr = (int(mean_right[0, 0]), int(mean_right[0, 1]))
    scale_factor = 30


    cv2.circle(rgb, left_cntr, 3, (255, 0, 255), 2)
    cv2.circle(rgb, right_cntr, 3, (255, 0, 255), 2)
    p1_left = (left_cntr[0] + 0.02 * eigenvectors_left[0, 0] * eigenvalues_left[0, 0], left_cntr[1] + 0.02 * eigenvectors_left[0, 1] * eigenvalues_left[0, 0])
    # p2_left = (left_cntr[0] - 0.02 * eigenvectors_left[1, 0] * eigenvalues_left[1, 0], left_cntr[1] - 0.02 * eigenvectors_left[1, 1] * eigenvalues_left[1, 0])

    p1_right = (right_cntr[0] + 0.02 * eigenvectors_right[0, 0] * eigenvalues_right[0, 0],
          right_cntr[1] + 0.02 * eigenvectors_right[0, 1] * eigenvalues_right[0, 0])
    # p2_right = (left_cntr[0] - 0.02 * eigenvectors_right[1, 0] * eigenvalues_right[1, 0],
    #       right_cntr[1] - 0.02 * eigenvectors_right[1, 1] * eigenvalues_right[1, 0])

    left_angle = angle_(left_cntr, p1_left)
    # drawAxis(rgb, left_cntr, p2_left, (255, 255, 0), 10)

    right_angle = angle_(right_cntr, p1_right)
    # drawAxis(rgb, right_cntr, p2_right, (255, 255, 0), 10)

    left_angle = np.rad2deg(left_angle)
    right_angle = np.rad2deg(right_angle)

    final_angle = (left_angle + right_angle)/2

    if left_angle < 0:
        left_angle = left_angle + 360

    if right_angle < 0:
        right_angle = right_angle + 360

    # cv2.putText(rgb, f"left_angle: {left_angle}", (0, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
    # cv2.putText(rgb, f"right_angle: {right_angle}", (0, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 2)


    mean_exflusion_h = int(np.mean(exflusion[0]))
    mean_exflusion_w = int(np.mean(exflusion[1]))

    y_rotated = mean_exflusion_h + int(np.sin(np.pi / -180 * final_angle)*600)
    x_rotated = mean_exflusion_w + int(np.cos(np.pi / -180 * final_angle)*600)

    # 기존 좌표와 이미지 끝에 교차된 점 까지를 이은 직선을 그립니다.
    cv2.line(rgb, (mean_exflusion_w, mean_exflusion_h), (x_rotated, y_rotated), (255, 0, 0))

    angle = atan2(float(mean_exflusion_h - y_rotated), float(mean_exflusion_w - x_rotated)) # angle in radians

    # create the arrow hooks
    mean_exflusion_w = x_rotated + 9 * cos(angle + pi / 4)
    mean_exflusion_h = y_rotated + 9 * sin(angle + pi / 4)
    cv2.line(img, (int(mean_exflusion_w), int(mean_exflusion_h)), (int(x_rotated), int(y_rotated)), (0, 255, 0), 10, cv2.LINE_AA)
    mean_exflusion_w = x_rotated + 9 * cos(angle - pi / 4)
    mean_exflusion_h = y_rotated + 9 * sin(angle - pi / 4)
    cv2.line(img, (int(mean_exflusion_w), int(mean_exflusion_h)), (int(x_rotated), int(y_rotated)), (0, 255, 0), 10, cv2.LINE_AA)


    # # create the arrow hooks
    # p[0] = q[0] + 9 * cos(angle + pi / 4)
    # p[1] = q[1] + 9 * sin(angle + pi / 4)
    # cv.line(img, (int(p[0]), int(p[1])), (int(q[0]), int(q[1])), colour, 1, cv.LINE_AA)
    # p[0] = q[0] + 9 * cos(angle - pi / 4)
    # p[1] = q[1] + 9 * sin(angle - pi / 4)
    # cv.line(img, (int(p[0]), int(p[1])), (int(q[0]), int(q[1])), colour, 1, cv.LINE_AA)

    if final_angle < 0:
        final_angle = final_angle + 360

    cv2.putText(rgb, f"final_angle: {final_angle}", (0, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 2)
    cv2.imwrite(f"./pca_angle/front/{img_path}", rgb)


    # while (1):
    #     cv2.imshow('image', rgb)
    #     k = cv2.waitKey(1) & 0xFF
    #
    #     if k == 27:
    #         break
    #
    # cv2.destroyAllWindows()

    return left_angle, right_angle

def front_pca_angle_direction_change(img, img_path, angle_list):
    # 이미지 좌표 y좌표 평행이동 및 y축 반전

    img_label_index = np.unique(img)

    rgb = visualization(img)

    exflusion = np.where(img == img_label_index[1])
    left_curve = np.where(img == img_label_index[2])
    right_curve = np.where(img == img_label_index[3])

    left_curve_h = left_curve[0]
    # y 축 기준으로 flip
    left_curve_h *= -1
    left_curve_w = left_curve[1]

    right_curve_h = right_curve[0]
    # y 축 기준으로 flip
    right_curve_h *= -1
    right_curve_w = right_curve[1]

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


    left_cntr = (int(mean_left[0, 0]), int(mean_left[0, 1]))
    right_cntr = (int(mean_right[0, 0]), int(mean_right[0, 1]))


    cv2.circle(rgb, left_cntr, 3, (255, 0, 255), 2)
    cv2.circle(rgb, right_cntr, 3, (255, 0, 255), 2)
    p1_left = (left_cntr[0] + 0.02 * eigenvectors_left[0, 0] * eigenvalues_left[0, 0], left_cntr[1] + 0.02 * eigenvectors_left[0, 1] * eigenvalues_left[0, 0])
    # p2_left = (left_cntr[0] - 0.02 * eigenvectors_left[1, 0] * eigenvalues_left[1, 0], left_cntr[1] - 0.02 * eigenvectors_left[1, 1] * eigenvalues_left[1, 0])

    p1_right = (right_cntr[0] + 0.02 * eigenvectors_right[0, 0] * eigenvalues_right[0, 0],
          right_cntr[1] + 0.02 * eigenvectors_right[0, 1] * eigenvalues_right[0, 0])
    # p2_right = (left_cntr[0] - 0.02 * eigenvectors_right[1, 0] * eigenvalues_right[1, 0],
    #       right_cntr[1] - 0.02 * eigenvectors_right[1, 1] * eigenvalues_right[1, 0])

    left_angle = angle_(left_cntr, p1_left)
    # drawAxis(rgb, left_cntr, p2_left, (255, 255, 0), 10)

    right_angle = angle_(right_cntr, p1_right)
    # drawAxis(rgb, right_cntr, p2_right, (255, 255, 0), 10)

    left_angle = np.rad2deg(left_angle)
    right_angle = np.rad2deg(right_angle)

    change_direction_hyperparameters = 180 # direction change = 180, not change = 0

    final_angle = (left_angle + right_angle)/2 + change_direction_hyperparameters

    if left_angle < 0:
        left_angle = left_angle + 360

    if right_angle < 0:
        right_angle = right_angle + 360

    # cv2.putText(rgb, f"left_angle: {left_angle}", (0, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
    # cv2.putText(rgb, f"right_angle: {right_angle}", (0, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 2)


    mean_exflusion_h = int(np.mean(exflusion[0]))
    mean_exflusion_w = int(np.mean(exflusion[1]))

    y_rotated = mean_exflusion_h + int(np.sin(np.pi / -180 * final_angle)*600)
    x_rotated = mean_exflusion_w + int(np.cos(np.pi / -180 * final_angle)*600)

    # 기존 좌표와 이미지 끝에 교차된 점 까지를 이은 직선을 그립니다.
    cv2.line(rgb, (mean_exflusion_w, mean_exflusion_h), (x_rotated, y_rotated), (0, 255, 0), 2, cv2.LINE_AA)

    angle = atan2(float(mean_exflusion_h - y_rotated), float(mean_exflusion_w - x_rotated)) # angle in radians

    # create the arrow hooks
    mean_exflusion_w = x_rotated + 9 * cos(angle + pi / 4)
    mean_exflusion_h = y_rotated + 9 * sin(angle + pi / 4)
    cv2.line(rgb, (int(mean_exflusion_w), int(mean_exflusion_h)), (int(x_rotated), int(y_rotated)), (0, 255, 0), 5, cv2.LINE_AA)
    mean_exflusion_w = x_rotated + 9 * cos(angle - pi / 4)
    mean_exflusion_h = y_rotated + 9 * sin(angle - pi / 4)
    cv2.line(rgb, (int(mean_exflusion_w), int(mean_exflusion_h)), (int(x_rotated), int(y_rotated)), (0, 255, 0), 5, cv2.LINE_AA)


    # # create the arrow hooks
    # p[0] = q[0] + 9 * cos(angle + pi / 4)
    # p[1] = q[1] + 9 * sin(angle + pi / 4)
    # cv.line(img, (int(p[0]), int(p[1])), (int(q[0]), int(q[1])), colour, 1, cv.LINE_AA)
    # p[0] = q[0] + 9 * cos(angle - pi / 4)
    # p[1] = q[1] + 9 * sin(angle - pi / 4)
    # cv.line(img, (int(p[0]), int(p[1])), (int(q[0]), int(q[1])), colour, 1, cv.LINE_AA)

    if final_angle < 0:
        final_angle = final_angle + 360

    cv2.putText(rgb, f"final_angle: {round(final_angle, 2)}", (0, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 2)
    cv2.imwrite(f"./pca_angle/front/{img_path}", rgb)

    angle_list.append(final_angle)

    # while (1):
    #     cv2.imshow('image', rgb)
    #     k = cv2.waitKey(1) & 0xFF
    #
    #     if k == 27:
    #         break
    #
    # cv2.destroyAllWindows()

    return left_angle, right_angle


def side_pca_angle_direction_change(img, img_path, angle_list):
    # 이미지 좌표 y좌표 평행이동 및 y축 반전

    img_label_index = np.unique(img)

    rgb = visualization(img)

    exflusion = np.where(img == img_label_index[1])
    left_curve = np.where(img == img_label_index[2])
    right_curve = np.where(img == img_label_index[3])

    left_curve_h = left_curve[0]
    # y 축 기준으로 flip
    left_curve_h *= -1
    left_curve_w = left_curve[1]

    right_curve_h = right_curve[0]
    # y 축 기준으로 flip
    right_curve_h *= -1
    right_curve_w = right_curve[1]

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


    left_cntr = (int(mean_left[0, 0]), int(mean_left[0, 1]))
    right_cntr = (int(mean_right[0, 0]), int(mean_right[0, 1]))


    cv2.circle(rgb, left_cntr, 3, (255, 0, 255), 2)
    cv2.circle(rgb, right_cntr, 3, (255, 0, 255), 2)
    p1_left = (left_cntr[0] + 0.02 * eigenvectors_left[0, 0] * eigenvalues_left[0, 0], left_cntr[1] + 0.02 * eigenvectors_left[0, 1] * eigenvalues_left[0, 0])
    # p2_left = (left_cntr[0] - 0.02 * eigenvectors_left[1, 0] * eigenvalues_left[1, 0], left_cntr[1] - 0.02 * eigenvectors_left[1, 1] * eigenvalues_left[1, 0])

    p1_right = (right_cntr[0] + 0.02 * eigenvectors_right[0, 0] * eigenvalues_right[0, 0],
          right_cntr[1] + 0.02 * eigenvectors_right[0, 1] * eigenvalues_right[0, 0])
    # p2_right = (left_cntr[0] - 0.02 * eigenvectors_right[1, 0] * eigenvalues_right[1, 0],
    #       right_cntr[1] - 0.02 * eigenvectors_right[1, 1] * eigenvalues_right[1, 0])

    left_angle = angle_(left_cntr, p1_left)
    # drawAxis(rgb, left_cntr, p2_left, (255, 255, 0), 10)

    right_angle = angle_(right_cntr, p1_right)
    # drawAxis(rgb, right_cntr, p2_right, (255, 255, 0), 10)

    left_angle = np.rad2deg(left_angle)
    right_angle = np.rad2deg(right_angle)

    change_direction_hyperparameters = 0 # direction change = 180, not change = 0

    final_angle = (left_angle + right_angle)/2 + change_direction_hyperparameters

    if left_angle < 0:
        left_angle = left_angle + 360

    if right_angle < 0:
        right_angle = right_angle + 360

    # cv2.putText(rgb, f"left_angle: {left_angle}", (0, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
    # cv2.putText(rgb, f"right_angle: {right_angle}", (0, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 2)


    mean_exflusion_h = int(np.mean(exflusion[0]))
    mean_exflusion_w = int(np.mean(exflusion[1]))

    y_rotated = mean_exflusion_h + int(np.sin(np.pi / -180 * final_angle)*600)
    x_rotated = mean_exflusion_w + int(np.cos(np.pi / -180 * final_angle)*600)

    # 기존 좌표와 이미지 끝에 교차된 점 까지를 이은 직선을 그립니다.
    cv2.line(rgb, (mean_exflusion_w, mean_exflusion_h), (x_rotated, y_rotated), (0, 255, 0), 2, cv2.LINE_AA)

    angle = atan2(float(mean_exflusion_h - y_rotated), float(mean_exflusion_w - x_rotated)) # angle in radians

    # create the arrow hooks
    mean_exflusion_w = x_rotated + 9 * cos(angle + pi / 4)
    mean_exflusion_h = y_rotated + 9 * sin(angle + pi / 4)
    cv2.line(rgb, (int(mean_exflusion_w), int(mean_exflusion_h)), (int(x_rotated), int(y_rotated)), (0, 255, 0), 5, cv2.LINE_AA)
    mean_exflusion_w = x_rotated + 9 * cos(angle - pi / 4)
    mean_exflusion_h = y_rotated + 9 * sin(angle - pi / 4)
    cv2.line(rgb, (int(mean_exflusion_w), int(mean_exflusion_h)), (int(x_rotated), int(y_rotated)), (0, 255, 0), 5, cv2.LINE_AA)


    # # create the arrow hooks
    # p[0] = q[0] + 9 * cos(angle + pi / 4)
    # p[1] = q[1] + 9 * sin(angle + pi / 4)
    # cv.line(img, (int(p[0]), int(p[1])), (int(q[0]), int(q[1])), colour, 1, cv.LINE_AA)
    # p[0] = q[0] + 9 * cos(angle - pi / 4)
    # p[1] = q[1] + 9 * sin(angle - pi / 4)
    # cv.line(img, (int(p[0]), int(p[1])), (int(q[0]), int(q[1])), colour, 1, cv.LINE_AA)

    if final_angle < 0:
        final_angle = final_angle + 360

    cv2.putText(rgb, f"final_angle: {round(final_angle, 2)}", (0, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 2)
    cv2.imwrite(f"./pca_angle/side/{img_path}", rgb)

    angle_list.append(final_angle)

    # while (1):
    #     cv2.imshow('image', rgb)
    #     k = cv2.waitKey(1) & 0xFF
    #
    #     if k == 27:
    #         break
    #
    # cv2.destroyAllWindows()

    return left_angle, right_angle



def side_pca_angle(img, img_path):

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

    mask_file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "pred_curve")

    image_list = os.listdir(mask_file_path)

    for image_path in image_list:

        angle_list = list()

        # img shape: h, w, c
        img = cv2.imread(os.path.join(mask_file_path, image_path), cv2.IMREAD_GRAYSCALE)

        # left or right 좌표 없을때 continue
        img_label_index = np.unique(img)

        if len(img_label_index) < 4:
            continue

        # 세점을 이용한 circle 각도 추출
        # left_angle, right_angle = three_points_circle_angle(img)

        # pca를 이용한 외곽선 점 데이터 고유벡터 선형 각도 추출
        # side
        # left_angle, right_angle = side_pca_angle(img, image_path)
        # front
        # left_angle, right_angle = front_pca_angle_case0234(img, image_path)

        left_angle, right_angle = side_pca_angle_direction_change(img, image_path, angle_list)

        # print(f"left_angle: {left_angle}\n")
        # print(f"right_angle: {right_angle}\n")






    #
    # # angle = atan2(eigenvectors[0, 1], eigenvectors[0, 0])  # orientation in radians
    #
    # # circle_left = plt.Circle((left_cntr[0], left_cntr[1]), 1, color='r', fill=False)
    # # circle_right = plt.Circle((right_cntr[0], right_cntr[1]), 1, color='r', fill=False)
    #
    # fig, ax = plt.subplots(figsize=(10, 10))
    #
    # ax.set_xlim([min(left_curve_array[:, 0] - 50), max(right_curve_array[:, 0]) + 50])
    # ax.set_ylim([min(left_curve_array[:, 1] - 50), max(right_curve_array[:, 1] + 50)])
    #
    # ax.scatter(left_curve_array[:, 0], left_curve_array[:, 1], s=2, c='red')
    # ax.scatter(right_curve_array[:, 0], right_curve_array[:, 1], s=2, c='blue')
    # # ax.add_patch(circle_left)
    # # ax.add_patch(circle_right)
    # # ax.plot([int(left_order_curve_array[0][0]), int(left_order_curve_array[0][0] + eigenvectors_left[0, 0] * scale_factor)], [int(left_order_curve_array[0][1]), int(left_order_curve_array[0][1] - eigenvectors_left[0, 1] * scale_factor)])
    # # ax.plot([int(right_order_curve_array[0][0]), int(right_order_curve_array[0][0] + eigenvectors_right[0, 0] * scale_factor)], [int(right_order_curve_array[0][1]), int(right_order_curve_array[0][1] - eigenvectors_right[0, 1] * scale_factor)])
    #
    # ax.text(left_cntr[0]+30, left_cntr[1]+30, f"left_angle: {left_angle}", fontsize=10)
    # ax.text(right_cntr[0]+30, right_cntr[1]+30, f"right_angle: {right_angle}", fontsize=10)
    #
    # if not os.path.exists("./pca_angle"):
    #     os.makedirs("./pca_angle")
    #
    # if not os.path.exists("./pca_angle/front"):
    #     os.makedirs("./pca_angle/front")
    #
    # fig.savefig(f"./pca_angle/front/{img_path}")

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