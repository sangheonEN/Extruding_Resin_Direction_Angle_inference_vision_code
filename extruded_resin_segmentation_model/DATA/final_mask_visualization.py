import os
import numpy as np
import cv2

base_path = os.path.dirname(os.path.abspath(__file__))
aug_data_path = os.path.join(base_path, "train_flip_aug_data")
save_path = os.path.join(base_path, "train_flip_aug_data", "mask_visualization")

if not os.path.exists(save_path):
    os.makedirs(save_path)

mask_path = os.path.join(base_path, "train_flip_aug_data", "mask")

mask_list = os.listdir(mask_path)

LEE_COLORMAP = [
    [0, 0, 0],
    [255, 255, 255],
    [255, 0, 0],
    [0, 0, 255]
]

LEE_COLORMAP_two_class = [
    [0, 0, 0],
    [255, 255, 255],
]


label_colors = LEE_COLORMAP
label_colors_two_class = LEE_COLORMAP_two_class


for mask_name in mask_list:

    mask = cv2.imread(os.path.join(mask_path, mask_name), cv2.IMREAD_GRAYSCALE)

    class_num = np.unique(mask).shape[0]

    if class_num < 3:

        r = mask.copy()
        g = mask.copy()
        b = mask.copy()

        for ll in range(0, 2):
            r[mask == ll] = label_colors_two_class[ll][0]
            g[mask == ll] = label_colors_two_class[ll][1]
            b[mask == ll] = label_colors_two_class[ll][2]

        rgb = np.zeros((mask.shape[0], mask.shape[1], 3))

        rgb[:, :, 0] = b.squeeze()
        rgb[:, :, 1] = g.squeeze()
        rgb[:, :, 2] = r.squeeze()

        cv2.imwrite(os.path.join(save_path, f"{mask_name}"), rgb)

    # visualization
    r = mask.copy()
    g = mask.copy()
    b = mask.copy()

    for ll in range(0, 4):
        r[mask == ll] = label_colors[ll][0]
        g[mask == ll] = label_colors[ll][1]
        b[mask == ll] = label_colors[ll][2]

    rgb = np.zeros((mask.shape[0], mask.shape[1], 3))

    rgb[:, :, 0] = b.squeeze()
    rgb[:, :, 1] = g.squeeze()
    rgb[:, :, 2] = r.squeeze()

    cv2.imwrite(os.path.join(save_path, f"{mask_name}"), rgb)