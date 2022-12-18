import cv2
import os
import numpy as np


LEE_COLORMAP = [
    [0, 0, 0],
    [0, 0, 255],
    [255, 255, 255],
    [255, 0, 0]
]

if __name__ == "__main__":

    img_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'test_data2')

    img_list = os.listdir(img_path)
    img_list = [file for file in img_list if file.endswith("png")]

    cnt = 0

    for i in img_list:

        img = cv2.imread(os.path.join(img_path, i), cv2.IMREAD_GRAYSCALE)
        print(np.unique(img))

        print(len(np.unique(img)))

        img[img == 4] = 1

        label_colours = LEE_COLORMAP

        r = img.copy()
        g = img.copy()
        b = img.copy()
        n_classes = len(np.unique(img))

        for ll in range(0, n_classes):
            r[img == ll] = label_colours[ll][0]
            g[img == ll] = label_colours[ll][1]
            b[img == ll] = label_colours[ll][2]
        rgb = np.zeros((img.shape[0], img.shape[1], 3))
        rgb[:, :, 0] = b.squeeze()
        rgb[:, :, 1] = g.squeeze()
        rgb[:, :, 2] = r.squeeze()

        cv2.imwrite(os.path.join(os.path.dirname(os.path.abspath(__file__)), '%04d.png' % cnt), rgb)
        cnt+=1



    # pixel value 1 is object and 2 is curve
    # img[img<=1.] = 0

    # object = img == 1.
    #
    # curve = img>=2.
    # curve = curve * 255.
