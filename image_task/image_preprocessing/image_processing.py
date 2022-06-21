class Angle_extract():
    def __init__(self) -> None:
        pass
    
    def proc(self):
        import cv2 as cv
        from math import atan2, cos, sin, sqrt, pi
        import numpy as np
        
        def drawAxis(img, p_, q_, color, scale):
            p = list(p_)
            q = list(q_)
            
            ## [visualization1]
            angle = atan2(p[1] - q[1], p[0] - q[0]) # angle in radians
            hypotenuse = sqrt((p[1] - q[1]) * (p[1] - q[1]) + (p[0] - q[0]) * (p[0] - q[0]))
            
            # Here we lengthen the arrow by a factor of scale
            q[0] = p[0] - scale * hypotenuse * cos(angle)
            q[1] = p[1] - scale * hypotenuse * sin(angle)
            cv.line(img, (int(p[0]), int(p[1])), (int(q[0]), int(q[1])), color, 1, cv.LINE_AA)
            
            # create the arrow hooks
            p[0] = q[0] + 9 * cos(angle + pi / 4)
            p[1] = q[1] + 9 * sin(angle + pi / 4)
            cv.line(img, (int(p[0]), int(p[1])), (int(q[0]), int(q[1])), color, 1, cv.LINE_AA)
            
            p[0] = q[0] + 9 * cos(angle - pi / 4)
            p[1] = q[1] + 9 * sin(angle - pi / 4)
            cv.line(img, (int(p[0]), int(p[1])), (int(q[0]), int(q[1])), color, 1, cv.LINE_AA)
            ## [visualization1]
        
        def getOrientation(pts, img):
            ## [pca]
            # Construct a buffer used by the pca analysis
            sz = len(pts)
            data_pts = np.empty((sz, 2), dtype=np.float64)
            for i in range(data_pts.shape[0]):
                data_pts[i,0] = pts[i,0,0]
                data_pts[i,1] = pts[i,0,1]
            
            # Perform PCA analysis
            mean = np.empty((0))
            mean, eigenvectors, eigenvalues = cv.PCACompute2(data_pts, mean)
            
            # Store the center of the object
            cntr = (int(mean[0,0]), int(mean[0,1]))
            ## [pca]
            
            ## [visualization]
            # Draw the principal components
            cv.circle(img, cntr, 3, (255, 0, 255), 2)
            p1 = (cntr[0] + 0.02 * eigenvectors[0,0] * eigenvalues[0,0], cntr[1] + 0.02 * eigenvectors[0,1] * eigenvalues[0,0])
            p2 = (cntr[0] - 0.02 * eigenvectors[1,0] * eigenvalues[1,0], cntr[1] - 0.02 * eigenvectors[1,1] * eigenvalues[1,0])
            drawAxis(img, cntr, p1, (255, 255, 0), 1)
            drawAxis(img, cntr, p2, (0, 0, 255), 1)
            
            angle = atan2(eigenvectors[0,1], eigenvectors[0,0]) # orientation in radians return atan(y / x)
            # angle2 = atan2(eigenvectors[1,1], eigenvectors[1,0]) # orientation in radians
            ## [visualization]
            
            # Label with the rotation angle
            label = "  Rotation Angle: " + str(-int(np.rad2deg(angle)) - 90) + " degrees"
            # label2 = "  Rotation Angle: " + str(-int(np.rad2deg(angle2)) - 90) + " degrees"
            # textbox = cv.rectangle(img, (cntr[0], cntr[1]-25), (cntr[0] + 250, cntr[1] + 10), (255,255,255), -1)
            cv.putText(img, label, (cntr[0], cntr[1]), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 1, cv.LINE_AA)
            # cv.putText(img, label2, (cntr[0]+100, cntr[1]+100), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 1, cv.LINE_AA)
            
            return angle
        
        if __name__ == "__main__":
            import cv2 as cv
            from math import atan2, cos, sin, sqrt, pi
            import numpy as np
            # Load the image
            img = cv.imread("./mask/Label_N.png")
            
            # Was the image there?
            if img is None:
                print("Error: File not found")
                exit(0)
            
            # cv.imshow('Input Image', img)
            
            # Convert image to grayscale
            gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
            
            # Convert image to binary
            _, bw = cv.threshold(gray, 50, 255, cv.THRESH_BINARY | cv.THRESH_OTSU)
            
            # Find all the contours in the thresholded image
            contours, _ = cv.findContours(bw, cv.RETR_LIST, cv.CHAIN_APPROX_NONE)
            
            for i, c in enumerate(contours):
            
                # Calculate the area of each contour
                area = cv.contourArea(c)
                
                # Ignore contours that are too small or too large
                # if area < 3700 or 100000 < area:
                #   continue
                
                # Draw each contour only for visualisation purposes
                cv.drawContours(img, contours, i, (0, 0, 255), 2)
                
                # Find the orientation of each shape
                getOrientation(c, img)
                
                # cv.imshow('Output Image', img)
                # cv.waitKey(0)
                # cv.destroyAllWindows()
            
            # Save the output image to the current directory
            cv.imwrite("output_img.jpg", img)




class Histogram():
    def __init__(self) -> None:
        pass

    def proc(self):
        import os
        import numpy as np
        import cv2
        import matplotlib.pyplot as plt

        base_path = os.path.abspath(os.path.dirname("__file__"))
        img_path = os.path.join(base_path, 'test_image')
        saver_path = os.path.join(base_path, 'his_equalization')

        folder_list = os.listdir(img_path)
        img_list = [file for file in folder_list if file.endswith('.png')]

        try:

            for img in img_list:
                src = cv2.imread(os.path.join(img_path, img))
                h, w, c = src.shape

                h1 = cv2.equalizeHist(src[:, :, 0])
                h2 = cv2.equalizeHist(src[:, :, 1])
                h3 = cv2.equalizeHist(src[:, :, 2])

                y = np.zeros((h, w, c), dtype=np.float32)

                y[:,:,0] = h1
                y[:,:,1] = h2
                y[:,:,2] = h3

                # rgb = cv2.cvtColor(y, cv2.COLOR_YCrCb2BGR)

                cv2.imwrite(os.path.join(saver_path, img), y)

        except Exception as e:
            print(e)



class Binary_proc():
    def __init__(self) -> None:
        pass

    def proc(self):

        import os
        import cv2
        import matplotlib.pyplot as plt
        import numpy as np
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D

        base_path = os.path.abspath(os.path.dirname(__file__))
        image_path = os.path.join(base_path, "image_data", "fps_24")
        save_path = os.path.join(base_path, "binary_data")

        image_name = "centering_coincidence_0327.png"
        crop_image_name = "centering_coincidence_0327_crop.png"

        def binary_threshold_seg(img, rgb_img):

            x, y, w, h = 154, 446, 141, 570
            crop = img[y: y+h, x:x+w]
            rgb_crop = rgb_img[y: y+h, x:x+w]

            # image GaussianBlur
            blur = cv2.GaussianBlur(crop,(5,5),0)

            # sharpening filter
            sharpening_mask1 = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
            sharpening_mask2 = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])

            sharpening1_img = cv2.filter2D(crop, -1, sharpening_mask1)
            sharpening2_img = cv2.filter2D(crop, -1, sharpening_mask2)

            """
            above filter
            below threshold method
            """

            # otsu threshold binary
            ret, blur_img_binary = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)

            ret, sharp1_img_binary = cv2.threshold(sharpening1_img, 0, 255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
            ret, sharp2_img_binary = cv2.threshold(sharpening2_img, 0, 255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)

            # canny edge detection
            edges = cv2.Canny(crop, 30, 70)

            # distance transform
            dist_trans = cv2.distanceTransform(crop, cv2.DIST_L2, 0)
            dist_trans = cv2.normalize(dist_trans, dist_trans, 0, 1.0, cv2.NORM_MINMAX)



            # Hough Transform
            minLineLength = 100
            maxLineGap = 1

            lines = cv2.HoughLinesP(blur_img_binary, 1, np.pi / 180, 15, minLineLength, maxLineGap)

            for x in range(0, len(lines)):
                for x1, y1, x2, y2 in lines[x]:
                    cv2.line(crop, (x1, y1), (x2, y2), (0, 128, 0), 2)


            # image save
            cv2.imwrite(os.path.join(save_path, "otsu_blur_"+image_name), blur_img_binary)
            cv2.imwrite(os.path.join(save_path, "otsu_sharp1_"+image_name), sharp1_img_binary)
            cv2.imwrite(os.path.join(save_path, "otsu_sharp2_"+image_name), sharp2_img_binary)
            cv2.imwrite(os.path.join(save_path, "canny_"+image_name), edges)
            cv2.imwrite(os.path.join(save_path, "distance_transform_"+image_name), dist_trans)
            cv2.imwrite(os.path.join(save_path, "blur_"+crop_image_name), blur)
            cv2.imwrite(os.path.join(save_path, "hough_"+crop_image_name), crop)

        def his(img):

            plt.figure()
            color = ('b', 'g', 'r')
            channels = cv2.split(img) # b, g, r
            for (ch, col) in zip(channels, color):
                histr = cv2.calcHist([ch], [0], None, [256], [0, 256])
                plt.plot(histr, color=col)
                plt.xlim([0, 256])
                plt.ylim([0, 20000])
                plt.savefig(os.path.join(save_path, "fig_"+image_name))

        def cluster_k_means(gray_img, rgb_img, img_name, visual_flag=False):

            x, y, w, h = 154, 446, 141, 570
            gray_img = gray_img[y: y+h, x:x+w]
            vectorized_gray = gray_img.reshape((-1, 1))
            vectorized_gray = np.float32(vectorized_gray)


            rgb_img = rgb_img[y: y+h, x:x+w]
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

                final_image = label_2 + label_3 + label_7

                cv2.imwrite(os.path.join(save_path, "cluster_2","clustering_" + img_name), final_image)

            print("zz")

        def blue_threshold(img):

            plt.figure()
            color = ('b', 'g', 'r')
            channels = cv2.split(img) # b, g, r
            for (ch, col) in zip(channels, color):
                if col == "b":
                    print(ch)


        if __name__ == "__main__":

            img_path_list = os.listdir(image_path)

            for img_name in img_path_list:

                # image load
                raw_img = cv2.imread(os.path.join(image_path, img_name))

                rgb_90 = cv2.rotate(raw_img, cv2.ROTATE_90_CLOCKWISE)
                gray_img = cv2.cvtColor(raw_img, cv2.COLOR_BGR2GRAY)
                gray_img_90 = cv2.rotate(gray_img, cv2.ROTATE_90_CLOCKWISE)
                # image crop
                gray_image_clone = gray_img_90.copy()
                rgb_image_clone = rgb_90.copy()

                # 실행 코드 함수 List
                # binary_threshold_seg(image_clone, raw_90)
                # his(raw_img)
                # blue_threshold(raw_90)
                cluster_k_means(gray_image_clone, rgb_image_clone, img_name, visual_flag=True)

            # rgb_img = cv2.cvtColor(rgb_img, cv2.COLOR_BGR2RGB)
            # r, g, b = cv2.split(rgb_img)
            # r = r.flatten()
            # g = g.flatten()
            # b = b.flatten()
            #
            # fig = plt.figure()
            # ax = Axes3D(fig)
            # ax.scatter(r, g, b)
            # plt.show()


class Circle_detec():
    def __init__(self) -> None:
        pass

    def proc(self):
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



class Trackbar_window():
    def __init__(self) -> None:
        pass


    def proc(self):
        
        """
        1. gray scale image에서 내가 원하는 threshold 값 이상이면 0 미만이면 255로 되도록 저장하고 출력
        단, 속도 빠르게
        """

        import cv2
        import numpy as np
        #
        # img_path = './his_equalization/case1.png'
        #
        # img = cv2.imread(img_path, flags=0)
        #
        # # ret, gray = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        #
        # cv2.imshow('zzz', img)
        #
        # img[img > 130] = 255
        # img[img <= 130] = 0
        #
        # cv2.imshow('zz', img)


        def onChange(pos):
            pass

        src = cv2.imread("./test_image/front_04_04_17_03_0954.png", cv2.IMREAD_GRAYSCALE)

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


