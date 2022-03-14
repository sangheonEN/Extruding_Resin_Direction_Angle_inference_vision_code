import os
import cv2
import matplotlib.pyplot as plt

base_path = os.path.abspath(os.path.dirname(__file__))
image_path = os.path.join(base_path, "image_data", "fps_24")
save_path = os.path.join(base_path, "binary_data")

image_name = "centering_coincidence_0327.png"
crop_image_name = "centering_coincidence_0327_crop.png"

def binary_threshold_seg(img):

    x, y, w, h = 154, 446, 141, 570
    crop = img[y: y+h, x:x+w]

    # image GaussianBlur
    blur = cv2.GaussianBlur(crop,(5,5),0)

    # image binary
    ret, img_binary = cv2.threshold(blur, 0, 255, cv2.THRESH_OTSU)

    # image save
    cv2.imwrite(os.path.join(save_path, image_name), img_binary)
    cv2.imwrite(os.path.join(save_path, crop_image_name), crop)

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
    # binary_threshold_seg(image_clone)
    # his(raw_img)
    blue_threshold(raw_90)

