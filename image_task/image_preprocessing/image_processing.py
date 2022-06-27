import cv2
from math import atan2, cos, sin, sqrt, pi
import numpy as np
import os
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


class Angle_extract:
    """
    self.proc: eigenvector and X_axis angle extract and img write.
    https://docs.opencv.org/3.4/d1/dee/tutorial_introduction_to_pca.html
    """
    def __init__(self, img_name_list, img_path, saver_path):
        self.img_name_list = img_name_list
        self.img_path = img_path
        self.saver_path = saver_path
    
    def angle_f(self, eigenvector):
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

    def getOrientation(self, pts, img, scale_factor):

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
        angle = self.angle_f(eigenvectors)

        # Store the center of the object (x, y)
        cntr = (int(mean[0,0]), int(mean[0,1]))
        ## [pca]
        
        ## [visualization]
        # Draw the principal components
        cv2.circle(img, cntr, 3, (255, 0, 255), 2)
        # 이미지상 y축은 아래로 양수니까 위쪽으로 선을 긋기 위해 eigenvector[0, 1] y값은 원점에 -로 더해주어 스케일 업 해준다.
        cv2.line(img, (int(cntr[0]), int(cntr[1])), (int(cntr[0] + eigenvectors[0,0] * scale_factor), int(cntr[1] - eigenvectors[0,1] * scale_factor)), (255, 255, 0), 1, cv2.LINE_AA)
        cv2.putText(img, f"Rotation_ Angle {str(angle)}", (cntr[0]+20, cntr[1]+20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 1, cv2.LINE_AA)


    def proc(self):
        try:
            for img_n in self.img_name_list:
                img = cv2.imread(os.path.join(self.img_path, img_n))

                # Was the image there?
                if img is None:
                    print("Error: File not found")
                    exit(0)
                            
                # Convert image to grayscale
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                
                # Convert image to binary
                _, bw = cv2.threshold(gray, 50, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
                
                # Find all the contours in the thresholded image
                contours, _ = cv2.findContours(bw, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
                
                for i, c in enumerate(contours):
                    
                    # Draw each contour only for visualisation purposes
                    cv2.drawContours(img, contours, i, (0, 0, 255), 2)
                    
                    # Find the orientation of each shape
                    self.getOrientation(c, img, scale_factor=50)
                    
                # Save the output image to the current directory
                cv2.imwrite(os.path.join(self.saver_path, img_n), img)

        except Exception as e:
            print(f"Error Descript: {e}")




class Histogram:
    def __init__(self, img_name_list, img_path, saver_path):
        self.img_name_list = img_name_list
        self.img_path = img_path
        self.saver_path = saver_path

    def histogram_distribution(self, img_n):
        
        src = cv2.imread(os.path.join(self.img_path, img_n))

        plt.figure()
        color = ('b', 'g', 'r')
        channels = cv2.split(src) # b, g, r
        for (ch, col) in zip(channels, color):
            histr = cv2.calcHist([ch], [0], None, [256], [0, 256])
            plt.plot(histr, color=col)
            plt.xlim([0, 256])
            plt.ylim([0, 20000])
            plt.savefig(os.path.join(self.saver_path, "his_"+img_n))

    def equalizehis(self, img_n):

        src = cv2.imread(os.path.join(self.img_path, img_n))
        h, w, c = src.shape

        h1 = cv2.equalizeHist(src[:, :, 0])
        h2 = cv2.equalizeHist(src[:, :, 1])
        h3 = cv2.equalizeHist(src[:, :, 2])

        y = np.zeros((h, w, c), dtype=np.float32)

        y[:,:,0] = h1
        y[:,:,1] = h2
        y[:,:,2] = h3

        # rgb = cv2.cvtColor(y, cv2.COLOR_YCrCb2BGR)

        cv2.imwrite(os.path.join(self.saver_path, img_n), y)


    def proc(self):
        try:
            for img_n in self.img_name_list:

                self.equalizehis(img_n)
                self.histogram_distribution(img_n)

        except Exception as e:
            print(f"Error Descript: {e}")



class Binary_proc:
    def __init__(self, img_name_list, img_path, saver_path):
        self.img_name_list = img_name_list
        self.img_path = img_path
        self.saver_path = saver_path


    def sobel_oper(self, img_n):

        # x, y, w, h = 300, 265, 674, 588

        img = cv2.imread(os.path.join(self.img_path, img_n), cv2.IMREAD_COLOR)
        img = cv2.GaussianBlur(img, (3, 3), 0)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # crop = gray[y: y+h, x:x+w]

        grad_x = cv2.Sobel(gray, ddepth=cv2.CV_16S, dx=1, dy=0, ksize=3, scale=1, delta=0, borderType=cv2.BORDER_DEFAULT)

        grad_y = cv2.Sobel(gray, ddepth=cv2.CV_16S, dx=0, dy=1, ksize=3, scale=1, delta=0, borderType=cv2.BORDER_DEFAULT)

        abs_grad_x = cv2.convertScaleAbs(grad_x)
        abs_grad_y = cv2.convertScaleAbs(grad_y)

        grad = cv2.addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0)

        if not os.path.exists(os.path.join(self.saver_path, "sobel")):
            os.makedirs(os.path.join(self.saver_path, "sobel"))

        cv2.imwrite(os.path.join(self.saver_path, 'sobel', "sobel_"+img_n), grad)


    def canny(self, img_n):

        img = cv2.imread(os.path.join(self.img_path, img_n), cv2.IMREAD_COLOR)

        # x, y, w, h = 300, 265, 674, 588
        # crop = gray[y: y+h, x:x+w]

        blur = cv2.GaussianBlur(img,(5,5),0)

        gray = cv2.cvtColor(blur, cv2.COLOR_BGR2GRAY)

        edges = cv2.Canny(gray, 30, 70)

        if not os.path.exists(os.path.join(self.saver_path, "canny")):
            os.makedirs(os.path.join(self.saver_path, "canny"))

        cv2.imwrite(os.path.join(self.saver_path, 'canny', "canny_"+img_n), edges)


    def sharpening(self, img_n):

        img = cv2.imread(os.path.join(self.img_path, img_n), cv2.IMREAD_COLOR)

        # x, y, w, h = 300, 265, 674, 588
        # crop = gray[y: y+h, x:x+w]

        blur = cv2.GaussianBlur(img,(5,5),0)

        gray = cv2.cvtColor(blur, cv2.COLOR_BGR2GRAY)

        # sharpening filter
        sharpening_mask1 = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
        sharpening_mask2 = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])

        sharpening1_img = cv2.filter2D(gray, -1, sharpening_mask1)
        sharpening2_img = cv2.filter2D(gray, -1, sharpening_mask2)

        ret, sharp1_img_binary = cv2.threshold(sharpening1_img, 0, 255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
        ret, sharp2_img_binary = cv2.threshold(sharpening2_img, 0, 255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)

        if not os.path.exists(os.path.join(self.saver_path, "sharpening")):
            os.makedirs(os.path.join(self.saver_path, "sharpening"))

        cv2.imwrite(os.path.join(self.saver_path, "sharpening", "sharp1_"+img_n), sharp1_img_binary)
        cv2.imwrite(os.path.join(self.saver_path, "sharpening", "sharp2_"+img_n), sharp2_img_binary)


    def hougn(self, img_n):

        img = cv2.imread(os.path.join(self.img_path, img_n), cv2.IMREAD_COLOR)

        # x, y, w, h = 300, 265, 674, 588
        # crop = gray[y: y+h, x:x+w]

        blur = cv2.GaussianBlur(img, (5, 5), 0)

        gray = cv2.cvtColor(blur, cv2.COLOR_BGR2GRAY)

        ret, blur_img_binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)


        # Hough Transform
        minLineLength = 100
        maxLineGap = 1

        lines = cv2.HoughLinesP(blur_img_binary, 1, np.pi / 180, 15, minLineLength, maxLineGap)

        for x in range(0, len(lines)):
            for x1, y1, x2, y2 in lines[x]:
                cv2.line(gray, (x1, y1), (x2, y2), (0, 128, 0), 2)

        if not os.path.exists(os.path.join(self.saver_path, "hougn")):
            os.makedirs(os.path.join(self.saver_path, "hougn"))

        cv2.imwrite(os.path.join(self.saver_path, 'hougn', "hougn_"+img_n), gray)


    # def binary_threshold_seg(self, img_n):
    #
    #     img = cv2.imread(os.path.join(self.img_path, img_n))
    #     gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    #
    #     x, y, w, h = 300, 265, 674, 588
    #     crop = gray[y: y+h, x:x+w]
    #
    #     # image GaussianBlur
    #     blur = cv2.GaussianBlur(crop,(5,5),0)
    #
    #     # sharpening filter
    #     sharpening_mask1 = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
    #     sharpening_mask2 = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    #
    #     sharpening1_img = cv2.filter2D(crop, -1, sharpening_mask1)
    #     sharpening2_img = cv2.filter2D(crop, -1, sharpening_mask2)
    #
    #     """
    #     above filter
    #     below threshold method
    #     """
    #
    #     # otsu threshold binary
    #     ret, blur_img_binary = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
    #
    #     ret, sharp1_img_binary = cv2.threshold(sharpening1_img, 0, 255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
    #     ret, sharp2_img_binary = cv2.threshold(sharpening2_img, 0, 255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
    #
    #     # canny edge detection
    #     edges = cv2.Canny(crop, 30, 70)
    #
    #     # # distance transform
    #     # dist_trans = cv2.distanceTransform(crop, cv2.DIST_L2, 0)
    #     # dist_trans = cv2.normalize(dist_trans, dist_trans, 0, 1.0, cv2.NORM_MINMAX)
    #
    #     # Hough Transform
    #     minLineLength = 100
    #     maxLineGap = 1
    #
    #     lines = cv2.HoughLinesP(blur_img_binary, 1, np.pi / 180, 15, minLineLength, maxLineGap)
    #
    #     for x in range(0, len(lines)):
    #         for x1, y1, x2, y2 in lines[x]:
    #             cv2.line(crop, (x1, y1), (x2, y2), (0, 128, 0), 2)
    #
    #     if not os.path.exists(os.path.join(self.saver_path, "otsu_blur")):
    #         os.makedirs(os.path.join(self.saver_path, "otsu_blur"))
    #         os.makedirs(os.path.join(self.saver_path, "otsu_sharp1"))
    #         os.makedirs(os.path.join(self.saver_path, "otsu_sharp2"))
    #         os.makedirs(os.path.join(self.saver_path, "canny"))
    #         os.makedirs(os.path.join(self.saver_path, "hough"))
    #
    #     # image save
    #     cv2.imwrite(os.path.join(self.saver_path, "otsu_blur_"+img_n), blur_img_binary)
    #     cv2.imwrite(os.path.join(self.saver_path, "otsu_sharp1_"+img_n), sharp1_img_binary)
    #     cv2.imwrite(os.path.join(self.saver_path, "otsu_sharp2_"+img_n), sharp2_img_binary)
    #     cv2.imwrite(os.path.join(self.saver_path, "canny_"+img_n), edges)
    #     cv2.imwrite(os.path.join(self.saver_path, "hough_"+img_n), crop)


    def proc(self):
        try:

            for img_name in self.img_name_list:

                self.canny(img_name)
                self.sharpening(img_name)
                self.hougn(img_name)
                self.sobel_oper(img_name)

        except Exception as e:
            print(f"Error Discript: {e}")


class Kmeans_cluster:
    def __init__(self, img_name_list, img_path, saver_path):
        self.img_name_list = img_name_list
        self.img_path = img_path
        self.saver_path = saver_path


    def cluster_k_means(self, img_n, visual_flag):

        img = cv2.imread(os.path.join(self.img_path, img_n))
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        x, y, w, h = 300, 265, 674, 588
        gray_img = gray[y: y+h, x:x+w]
        vectorized_gray = gray_img.reshape((-1, 1))
        vectorized_gray = np.float32(vectorized_gray)


        rgb_img = img[y: y+h, x:x+w]
        vectorized_rgb = rgb_img.reshape((-1, 3))
        vectorized_rgb = np.float32(vectorized_rgb)

        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)

        """
        cv2.kmeans(sample, nclusters(K), criteria, attempts, flags)

        3. criteria: It is the iteration termination criteria. When this criterion is satisfied, the algorithm iteration stops. 
        Actually, it should be a tuple of 3 parameters. They are `( type, max_iter, epsilon )`
        Type of termination criteria. It has 3 flags as below:

        cv.TERM_CRITERIA_EPS — stop the algorithm iteration if specified accuracy, epsilon, is reached.
        cv.TERM_CRITERIA_MAX_ITER — stop the algorithm after the specified number of iterations, max_iter.
        cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER — stop the iteration when any of the above condition is met.
        
        """
        K = 10
        attempts = 10

        ret_rgb, label_rgb, center_rgb = cv2.kmeans(vectorized_rgb, K, None, criteria, attempts, cv2.KMEANS_PP_CENTERS)
        ret_gray, label_gray, center_gray = cv2.kmeans(vectorized_gray, K, None, criteria, attempts, cv2.KMEANS_PP_CENTERS)

        #rgb
        center_rgb = np.uint8(center_rgb)
        res_rgb = center_rgb[label_rgb.flatten()]
        result_image_rgb = res_rgb.reshape(rgb_img.shape)

        if visual_flag:

            figure_size = 15
            plt.figure(figsize=(figure_size, figure_size))
            # each label image extract
            for i in range(K):
                globals()["label_{}".format(i+1)] = np.zeros(label_rgb.shape)
                globals()["label_{}".format(i+1)][np.where(label_rgb == i)] = 1
                globals()["label_{}".format(i+1)] = globals()["label_{}".format(i+1)].reshape((rgb_img.shape[0], rgb_img.shape[1]))*255.
                plt.subplot(1, K, i+1), plt.imshow(globals()["label_{}".format(i+1)])
                plt.title(f'Segmented Image when {i+1}/{K}'), plt.xticks([]), plt.yticks([])
            plt.show()

        else:
            for i in range(K):
                globals()["label_{}".format(i+1)] = np.zeros(label_rgb.shape)
                globals()["label_{}".format(i+1)][np.where(label_rgb == i)] = 1
                globals()["label_{}".format(i+1)] = globals()["label_{}".format(i+1)].reshape((rgb_img.shape[0], rgb_img.shape[1]))*255.

            k1_image = label_1
            k2_image = label_2
            k3_image = label_3
            k4_image = label_4
            k5_image = label_5
            k6_image = label_6
            k7_image = label_7
            k8_image = label_8
            k9_image = label_9
            k10_image = label_10
            final_image = label_7 + label_8 + label_10

            # final_image = label_7 + label_9 + label_10

            if not os.path.exists(os.path.join(self.saver_path, "clustering")):
                os.makedirs(os.path.join(self.saver_path, "clustering"))

            cv2.imwrite(os.path.join(self.saver_path, "clustering", f"clustering_k{1}_" + img_n), k1_image)
            cv2.imwrite(os.path.join(self.saver_path, "clustering", f"clustering_k{2}_" + img_n), k2_image)
            cv2.imwrite(os.path.join(self.saver_path, "clustering", f"clustering_k{3}_" + img_n), k3_image)
            cv2.imwrite(os.path.join(self.saver_path, "clustering", f"clustering_k{4}_" + img_n), k4_image)
            cv2.imwrite(os.path.join(self.saver_path, "clustering", f"clustering_k{5}_" + img_n), k5_image)
            cv2.imwrite(os.path.join(self.saver_path, "clustering", f"clustering_k{6}_" + img_n), k6_image)
            cv2.imwrite(os.path.join(self.saver_path, "clustering", f"clustering_k{7}_" + img_n), k7_image)
            cv2.imwrite(os.path.join(self.saver_path, "clustering", f"clustering_k{8}_" + img_n), k8_image)
            cv2.imwrite(os.path.join(self.saver_path, "clustering", f"clustering_k{9}_" + img_n), k9_image)
            cv2.imwrite(os.path.join(self.saver_path, "clustering", f"clustering_k{10}_" + img_n), k10_image)
            cv2.imwrite(os.path.join(self.saver_path, "clustering", f"clustering_final_" + img_n), final_image)


    def proc(self):
        try:

            for img_name in self.img_name_list:

                self.cluster_k_means(img_name, visual_flag=False)

        except Exception as e:
            print(f"Error Discript: {e}")



class Circle_detec:
    def __init__(self, img_name_list, img_path, saver_path):
        self.img_name_list = img_name_list
        self.img_path = img_path
        self.saver_path = saver_path

    
    def cc_detec(self, img_n):

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

        img = cv2.imread(os.path.join(self.img_path, img_n))

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
        

    def proc(self):

        try:
            
            for img_name in self.img_name_list:

                self.cc_detec(img_name)
        
        except Exception as e:
            print(f"Error Discript: {e}")


class Trackbar_window:
    def __init__(self, img_name_list, img_path, saver_path):
        self.img_name_list = img_name_list
        self.img_path = img_path
        self.saver_path = saver_path        

    
    def trackbar_window(self, img_n):

        def onChange(pos):
            pass

        src = cv2.imread(os.path.join(self.img_path, img_n), cv2.IMREAD_GRAYSCALE)

        cv2.namedWindow("Trackbar Windows")

        cv2.createTrackbar("threshold", "Trackbar Windows", 0, 255, onChange)
        cv2.createTrackbar("maxValue", "Trackbar Windows", 0, 255, lambda x : x)

        cv2.setTrackbarPos("threshold", "Trackbar Windows", 127)
        cv2.setTrackbarPos("maxValue", "Trackbar Windows", 255)

        while cv2.waitKey(1) != ord('q'):

            thresh = cv2.getTrackbarPos("threshold", "Trackbar Windows")
            maxval = cv2.getTrackbarPos("maxValue", "Trackbar Windows")

            _, binary = cv2.threshold(src, thresh, maxval, cv2.THRESH_BINARY)

            cv2.imshow("Trackbar Windows", binary)


        cv2.waitKey(0)
        cv2.destroyAllWindows()


    def proc(self):
        
        try:

            for img_name in self.img_name_list:

                self.trackbar_window(img_name)

        except Exception as e:
            print(e)




if __name__ == "__main__":

    base_path = os.path.abspath(os.path.dirname("__file__"))

    if not os.path.exists(os.path.join(base_path, 'before_processing')):
        os.makedirs(os.path.join(base_path, 'before_processing'))

    img_path = os.path.join(base_path, 'before_processing')

    if not os.path.exists(os.path.join(base_path, 'after_processing')):
        os.makedirs(os.path.join(base_path, 'after_processing'))

    saver_path = os.path.join(base_path, 'after_processing')

    folder_list = os.listdir(img_path)
    img_list = [file for file in folder_list if file.endswith('.png')]

    # # image_path, image_list, save_path

    # binary_proc = Binary_proc(img_list, img_path, saver_path)
    # binary_proc.proc()

    kmeans_cluster_1 = Kmeans_cluster(img_list, img_path, saver_path)
    kmeans_cluster_1.proc()


    print("End Processing")


