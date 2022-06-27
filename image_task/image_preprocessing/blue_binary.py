import cv2
import os
from itertools import combinations
import numpy as np
from math import atan2


def kapur_threshold(image):
    """ Runs the Kapur's threshold algorithm.
    Reference:
    Kapur, J. N., P. K. Sahoo, and A. K. C.Wong. ‘‘A New Method for Gray-Level
    Picture Thresholding Using the Entropy of the Histogram,’’ Computer Vision,
    Graphics, and Image Processing 29, no. 3 (1985): 273–285.
    @param image: The input image
    @type image: ndarray
    @return: The estimated threshold
    @rtype: int
    """
    hist, _ = np.histogram(image, bins=range(256), density=True)
    c_hist = hist.cumsum()
    c_hist_i = 1.0 - c_hist

    # To avoid invalid operations regarding 0 and negative values.
    c_hist[c_hist <= 0] = 1
    c_hist_i[c_hist_i <= 0] = 1

    c_entropy = (hist * np.log(hist + (hist <= 0))).cumsum()
    b_entropy = -c_entropy / c_hist + np.log(c_hist)

    c_entropy_i = c_entropy[-1] - c_entropy
    f_entropy = -c_entropy_i / c_hist_i + np.log(c_hist_i)

    return np.argmax(b_entropy + f_entropy)


def _get_regions_entropy(hist, c_hist, thresholds):
    """Get the total entropy of regions for a given set of thresholds"""

    total_entropy = 0
    for i in range(len(thresholds) - 1):
        # Thresholds
        t1 = thresholds[i] + 1
        t2 = thresholds[i + 1]

        # print(thresholds, t1, t2)

        # Cumulative histogram
        hc_val = c_hist[t2] - c_hist[t1 - 1]

        # Normalized histogram
        h_val = hist[t1:t2 + 1] / hc_val if hc_val > 0 else 1

        # entropy
        entropy = -(h_val * np.log(h_val + (h_val <= 0))).sum()

        # Updating total entropy
        total_entropy += entropy

    return total_entropy


def _get_thresholds(hist, c_hist, nthrs):
    """Get the thresholds that maximize the entropy of the regions
    @param hist: The normalized histogram of the image
    @type hist: ndarray
    @param c_hist: The cummuative normalized histogram of the image
    @type c_hist: ndarray
    @param nthrs: The number of thresholds
    @type nthrs: int
    """
    # Thresholds combinations
    thr_combinations = combinations(range(255), nthrs)

    max_entropy = 0
    opt_thresholds = None

    # Extending histograms for convenience
    # hist = np.append([0], hist)
    c_hist = np.append(c_hist, [0])

    for thresholds in thr_combinations:
        # Extending thresholds for convenience
        e_thresholds = [-1]
        e_thresholds.extend(thresholds)
        e_thresholds.extend([len(hist) - 1])

        # Computing regions entropy for the current combination of thresholds
        regions_entropy = _get_regions_entropy(hist, c_hist, e_thresholds)

        if regions_entropy > max_entropy:
            max_entropy = regions_entropy
            opt_thresholds = thresholds

    return opt_thresholds


def kapur_multithreshold(image, nthrs):
    """ Runs the Kapur's multi-threshold algorithm.
    Reference:
    Kapur, J. N., P. K. Sahoo, and A. K. C.Wong. ‘‘A New Method for Gray-Level
    Picture Thresholding Using the Entropy of the Histogram,’’ Computer Vision,
    Graphics, and Image Processing 29, no. 3 (1985): 273–285.
    @param image: The input image
    @type image: ndarray
    @param nthrs: The number of thresholds
    @type nthrs: int
    @return: The estimated threshold
    @rtype: int
    """
    # Histogran
    hist, _ = np.histogram(image, bins=range(256), density=True)

    # Cumulative histogram
    c_hist = hist.cumsum()

    return _get_thresholds(hist, c_hist, nthrs)


def getOrientation(pts, img, scale_factor):

    ## [pca]
    # Construct a buffer used by the pca analysis
    sz = len(pts)
    data_pts = np.empty((sz, 2), dtype=np.float64)
    for i in range(data_pts.shape[0]):
        data_pts[i,0] = pts[i,0,0]
        data_pts[i,1] = pts[i,0,1]
        
    # Perform PCA analysis
    """
    cv2.PCACompute2는 data의 공분산 Matrix를 계산하고 이 공분산 Matrix의 eigenvalue, eigenvector를 출력한다.
    그리고 data의 mean 값도 출력한다. (Location 정보로 사용가능)
    eigenvalue가 큰 것부터 정렬되고 eigenvector와 매칭되어 출력됨.
    즉, eigenvalue[0, 0], eigenvector[0, 0] 이 x의 주성분이고 eigenvalue[0, 1], eigenvector[0, 1]이 y의 주성분이다.
    """
    mean = np.empty((0))
    mean, eigenvectors, eigenvalues = cv2.PCACompute2(data_pts, mean)

    # angle calculation by atan2(y, x)
    angle = angle_f(eigenvectors)

    # Store the center of the object (x, y)
    cntr = (int(mean[0,0]), int(mean[0,1]))
    ## [pca]
        
    ## [visualization]
    # Draw the principal components
    cv2.circle(img, cntr, 3, (255, 0, 255), 2)
    # 이미지상 y축은 아래로 양수니까 위쪽으로 선을 긋기 위해 eigenvector[0, 1] y값은 원점에 -로 더해주어 스케일 업 해준다.
    cv2.line(img, (int(cntr[0]), int(cntr[1])), (int(cntr[0] + eigenvectors[0,0] * scale_factor), int(cntr[1] - eigenvectors[0,1] * scale_factor)), (255, 255, 0), 1, cv2.LINE_AA)
    cv2.putText(img, f"Rotation_ Angle {str(angle)}", (cntr[0]+20, cntr[1]+20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 1, cv2.LINE_AA)



def angle_f(eigenvector):
    # angle calculation
    # atan2(y, x) 에서 예각이면 오른쪽 방향, 둔각이면 왼쪽 방향으로 설정. atan2 각도 설명 ppt참고.

    # y축 기준 역전시키기. eigenvector y 값이 양수 일때  
    if eigenvector[0, 1] > 0:
        eigenvector[0, 0] = -eigenvector[0, 0]
    elif eigenvector[0, 1] < 0:
        eigenvector[0, 1] = -eigenvector[0, 1]
    else:
        pass

    angle = atan2(eigenvector[0, 1], eigenvector[0, 0])


    return int(np.rad2deg(angle))




if __name__ == "__main__":

    base_path = os.path.abspath(os.path.dirname("__file__"))

    if not os.path.exists(os.path.join(base_path, 'after_processing')):
        os.makedirs(os.path.join(base_path, 'after_processing'))

    saver_path = os.path.join(base_path, 'after_processing')

    img_path = os.path.join(base_path, 'test_data')

    img_list = os.listdir(img_path)

    img_list = [file for file in img_list if file.endswith('.png')]


    for img_n in img_list:

        img = cv2.imread(os.path.join(img_path, img_n), cv2.IMREAD_COLOR)
        # cv2.IMREAD_COLOR: b, g, r
        h, w, c = img.shape

        blue = np.zeros((h, w, 1), dtype=np.float32)

        blue = img[:,:,0]

        x, y, w, h = 520, 520, 320, 180
        crop = blue[y : y+h, x : x+w]

        threshold_value = kapur_multithreshold(crop, 2)

        ret, final_image = cv2.threshold(crop, threshold_value[1], 255, cv2.THRESH_BINARY)

        contours, hierarchy = cv2.findContours(final_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

        initial_contour_area = 0.0

        idx = 0

        for contour in contours:
            
            area = cv2.contourArea(contour)

            if area >= initial_contour_area:
                initial_contour_area = area
                best_contour = contour
            
            idx += 1
        
        final_image = cv2.cvtColor(final_image, cv2.COLOR_GRAY2BGR)

        cv2.drawContours(final_image, best_contour, -1, color=(0, 0, 255), thickness=3)

        # cv2.imshow("contour", final_image)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

        getOrientation(best_contour, final_image, scale_factor=10)

        cv2.imwrite(os.path.join(saver_path, img_n), final_image)




