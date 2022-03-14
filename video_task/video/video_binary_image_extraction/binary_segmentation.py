import os
import cv2
import matplotlib.pyplot as plt
import numpy as np

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

    # otsu threshold binary
    ret, img_binary = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)

    # canny edge detection
    edges = cv2.Canny(crop, 30, 70)

    # distance transform
    dist_trans = cv2.distanceTransform(crop, cv2.DIST_L2, 0)
    dist_trans = cv2.normalize(dist_trans, dist_trans, 0, 1.0, cv2.NORM_MINMAX)

    # Hough Transform
    minLineLength = 100
    maxLineGap = 1

    lines = cv2.HoughLinesP(img_binary, 1, np.pi / 180, 15, minLineLength, maxLineGap)

    for x in range(0, len(lines)):
        for x1, y1, x2, y2 in lines[x]:
            cv2.line(crop, (x1, y1), (x2, y2), (0, 128, 0), 2)


    # image save
    cv2.imwrite(os.path.join(save_path, "otsu_"+image_name), img_binary)
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
    raw_90 = cv2.rotate(raw_img, cv2.ROTATE_90_CLOCKWISE)
    gray_img = cv2.cvtColor(raw_img, cv2.COLOR_BGR2GRAY)
    img_90 = cv2.rotate(gray_img, cv2.ROTATE_90_CLOCKWISE)
    # image crop
    image_clone = img_90.copy()



    # 실행 코드 함수 List
    binary_threshold_seg(image_clone, raw_90)
    # his(raw_img)
    # blue_threshold(raw_90)

