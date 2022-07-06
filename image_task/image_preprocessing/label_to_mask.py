import numpy as np
import cv2
import os

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

    img_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'test_data', 'Label_14.png')

    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

    print(np.unique(img))

    # pixel value 1 is object and 2 is curve
    # img[img<=1.] = 0

    object = img == 1.

    curve = img>=2.
    curve = curve * 255.

    # cv2.imshow("zz", curve)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    cv2.imwrite("./case_cp_sample.png", curve)

    curve_h, curve_w = np.where(curve == 255.)

    left_curve_idx = np.argwhere(curve_w <= 680)

    left_w = [curve_w[idx] for idx in left_curve_idx]
    left_h = [curve_h[idx] for idx in left_curve_idx]

    x_points, y_points = extract_three_points(left_w, left_h)

    print("zz")

































# cv2.imshow("zz", curve)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# w, h = curve.shape[0], curve.shape[1]

# arr = np.zeros((w, h, 3))

# arr[:,:,0] = object
# arr[:,:,1] = object
# arr[:,:,2] = object
# arr[:,:,2] = curve

# cv2.imwrite("./a.png", arr)


# # img = img * 255.

# # cv2.imshow('zz', img)
# # cv2.waitKey(0)
# # cv2.destroyAllWindows()

