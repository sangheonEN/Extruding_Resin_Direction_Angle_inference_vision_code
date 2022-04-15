import os
import numpy as np
import cv2

base_path = os.path.abspath(os.path.dirname("__file__"))
circle_path = os.path.join(base_path, 'his_equalization', 'scaling')
test_image = 'gt_test.png'
test_image_path = os.path.join(circle_path, test_image)

if __name__ == "__main__":
    # img read and resize
    img = cv2.imread(test_image_path)
    img = cv2.resize(img, dsize=(1000, 1000), interpolation=cv2.INTER_LINEAR)
    # cv2.imwrite('./zz.png', img)

    # circle detection

    """
검출 방법은 항상 2단계 허프 변환 방법(21HT, 그레이디언트)만 사용합니다.

resolution_ratio은 원의 중심을 검출하는 데 사용되는 누산 평면의 해상도를 의미합니다.

인수를 1로 지정할 경우 입력한 이미지와 동일한 해상도를 가집니다. 즉, 입력 이미지 너비와 높이가 동일한 누산 평면이 생성됩니다.

또한 인수를 2로 지정하면 누산 평면의 해상도가 절반으로 줄어 입력 이미지의 크기와 반비례합니다.

each_circle_min_distance는 일차적으로 검출된 원과 원 사이의 최소 거리입니다. 이 값은 원이 여러 개 검출되는 것을 줄이는 역할을 합니다.

canny_threshold은 허프 변환에서 자체적으로 캐니 엣지를 적용하게 되는데, 이때 사용되는 상위 임곗값을 의미합니다.

하위 임곗값은 자동으로 할당되며, 상위 임곗값의 절반에 해당하는 값을 사용합니다.

mid_threshold은 그레이디언트 방법에 적용된 중심 히스토그램(누산 평면)에 대한 임곗값입니다. 이 값이 낮을 경우 더 많은 원이 검출됩니다.

최소 반지름과 최대 반지름은 검출될 원의 반지름 범위입니다. 0을 입력할 경우 검출할 수 있는 반지름에 제한 조건을 두지 않습니다.

최소 반지름과 최대 반지름에 각각 0을 입력할 경우 반지름을 고려하지 않고 검출하며, 최대 반지름에 음수를 입력할 경우 검출된 원의 중심만 반환합니다.
gt param
    # small circle
    resolution_ratio_s = 1
    each_circle_min_distance_s = 2
    canny_threshold_s = 101
    mid_threshold_s = 25
    minRadius_s = 0
    maxRadius_s = 95

    """
    # small circle
    resolution_ratio_s = 1
    each_circle_min_distance_s = 10
    canny_threshold_s = 155
    mid_threshold_s = 10
    minRadius_s = 10
    maxRadius_s = 15

    # big circle
    resolution_ratio_b = 1
    each_circle_min_distance_b = 250
    canny_threshold_b = 101
    mid_threshold_b = 20
    minRadius_b = 50
    maxRadius_b = 100

    img_num = 0

    dst_s = img.copy()
    dst_b = img.copy()

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.medianBlur(gray, 3)
    circles_small = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, resolution_ratio_s, each_circle_min_distance_s, param1=canny_threshold_s, param2=mid_threshold_s, minRadius=minRadius_s, maxRadius=maxRadius_s)
    circles_big = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, resolution_ratio_b, each_circle_min_distance_b, param1=canny_threshold_b, param2=mid_threshold_b, minRadius=minRadius_b, maxRadius=maxRadius_b)
    for i in circles_big[0]:
        # dst = img.copy()
        cv2.circle(dst_b, (i[0], i[1]), i[2], (255, 255, 255), 1)

        # cv2.imwrite(os.path.join('./circle_detection_result/%04d.png' %img_num), dst)

        # img_num += 1

        # if img_num == 1 or img_num == 50:
        #     print(i)

    for i in circles_small[0]:
        # dst = img.copy()
        cv2.circle(dst_s, (i[0], i[1]), i[2], (255, 255, 255), 1)

        # cv2.imwrite(os.path.join('./circle_detection_result/%04d.png' %img_num), dst)

        # img_num += 1

        # if img_num == 1 or img_num == 50:
        #     print(i)

    cv2.imshow("small", dst_s)
    cv2.imshow("big", dst_b)
    cv2.waitKey()
    cv2.destroyAllWindows()
