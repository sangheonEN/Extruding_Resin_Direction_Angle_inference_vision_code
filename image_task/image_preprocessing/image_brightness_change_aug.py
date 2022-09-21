import imgaug.augmenters as iaa
import cv2
import os
import numpy as np

# aug1 = iaa.WithBrightnessChannels(iaa.Add((-50, 50)))
#
# aug2 = iaa.WithBrightnessChannels(
#     iaa.Add((-50, 50)), to_colorspace=[iaa.CSPACE_Lab, iaa.CSPACE_HSV])
#
# aug3 = iaa.AddToBrightness((-30, 30))


if __name__ == "__main__":

    img_list = os.listdir('./img_brightness_aug_input_img')

    seq = iaa.Sequential([
        iaa.WithBrightnessChannels(iaa.Add((-50, 50)))
    ])

    for i in img_list:
        img = cv2.imread(f'./img_brightness_aug_input_img/{i}', cv2.IMREAD_COLOR)
        img = np.expand_dims(img, axis=0)
        img_aug = seq(images=img)
        cv2.imwrite(f"./img_brightness_aug_output_img/{i}", np.squeeze(img_aug))