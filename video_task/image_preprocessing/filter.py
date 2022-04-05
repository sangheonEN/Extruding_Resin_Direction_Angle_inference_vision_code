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

    #
    # y = np.zeros((h, w), dtype=np.float)
    # cr = np.zeros((h, w), dtype=np.float)
    # cb = np.zeros((h, w), dtype=np.float)
    #
    # cv2.COLOR_YCrCb2BGR
