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

def cluster_k_means(gray_img, rgb_img):

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
    K = 50
    attempts = 10


    ret_rgb, label_rgb, center_rgb = cv2.kmeans(vectorized_rgb, K, None, criteria, attempts, cv2.KMEANS_PP_CENTERS)
    ret_gray, label_gray, center_gray = cv2.kmeans(vectorized_gray, K, None, criteria, attempts, cv2.KMEANS_PP_CENTERS)

    #rgb
    center_rgb = np.uint8(center_rgb)
    res_rgb = center_rgb[label_rgb.flatten()]
    result_image_rgb = res_rgb.reshape(rgb_img.shape)

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

    # label_1[np.where(label_rgb==0)] = 1
    # label_2[np.where(label_rgb==1)] = 1
    # label_3[np.where(label_rgb==2)] = 1
    # label_4[np.where(label_rgb==3)] = 1
    # label_5[np.where(label_rgb==4)] = 1

    # figure_size = 15
    # plt.figure(figsize=(figure_size, figure_size))
    # plt.subplot(1, 5, 1), plt.imshow(label_1.reshape((rgb_img.shape[0], rgb_img.shape[1]))*255.)
    # plt.title(f'Segmented Image when 1/{K}'), plt.xticks([]), plt.yticks([])
    # plt.subplot(1, 5, 2), plt.imshow(label_2.reshape((rgb_img.shape[0], rgb_img.shape[1]))*255.)
    # plt.title(f'Segmented Image when 2/{K}'), plt.xticks([]), plt.yticks([])
    # plt.subplot(1, 5, 3), plt.imshow(label_3.reshape((rgb_img.shape[0], rgb_img.shape[1]))*255.)
    # plt.title(f'Segmented Image when 3/{K}'), plt.xticks([]), plt.yticks([])
    # plt.subplot(1, 5, 4), plt.imshow(label_4.reshape((rgb_img.shape[0], rgb_img.shape[1]))*255.)
    # plt.title(f'Segmented Image when 4/{K}'), plt.xticks([]), plt.yticks([])
    # plt.subplot(1, 5, 5), plt.imshow(label_5.reshape((rgb_img.shape[0], rgb_img.shape[1]))*255.)
    # plt.title(f'Segmented Image when 5/{K}'), plt.xticks([]), plt.yticks([])
    # plt.show()

    cv2.imwrite(os.path.join(save_path, "cluster","best_label"+str(K)+"_"+image_name), label_3.reshape((rgb_img.shape[0], rgb_img.shape[1]))*255.)

    cv2.imwrite(os.path.join(save_path, "cluster","clustering_rgb_ed_"+str(K)+"_"+image_name), result_image_rgb)
    cv2.imwrite(os.path.join(save_path, "rgb"+str(K)+"_"+image_name), rgb_img)

    # gray
    center_gray = np.uint8(center_gray)
    res_gray = center_gray[label_gray.flatten()]
    result_image_gray = res_gray.reshape(gray_img.shape)

    cv2.imwrite(os.path.join(save_path, "cluster","clustering_gray_ed_"+str(K)+"_"+image_name), result_image_gray)
    cv2.imwrite(os.path.join(save_path, "cluster","gray"+str(K)+"_"+image_name), gray_img)









def blue_threshold(img):

    plt.figure()
    color = ('b', 'g', 'r')
    channels = cv2.split(img) # b, g, r
    for (ch, col) in zip(channels, color):
        if col == "b":
            print(ch)


if __name__ == "__main__":

    # image load
    raw_img = cv2.imread(os.path.join(image_path, image_name))
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
    cluster_k_means(gray_image_clone, rgb_image_clone)

