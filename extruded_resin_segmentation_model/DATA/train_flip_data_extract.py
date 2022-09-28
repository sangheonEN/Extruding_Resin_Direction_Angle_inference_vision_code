import cv2
import os
import albumentations as A
import numpy as np


def mask_preprocessing(mask, horizontal_mask, vertical_mask, img_name):
    # curve 일관성 있게 라벨링 하기 위한 preprocessing

    case = img_name.split("_")[0]

    if case == "case0":
        temp = 4
        switch_class_1 = 2
        swtich_class_2 = 3

        horizontal_mask[horizontal_mask == switch_class_1] = temp
        horizontal_mask[horizontal_mask == swtich_class_2] = switch_class_1
        horizontal_mask[horizontal_mask == temp] = swtich_class_2

    if case == "case2":
        temp = 4
        switch_class_1 = 2
        swtich_class_2 = 3

        horizontal_mask[horizontal_mask == switch_class_1] = temp
        horizontal_mask[horizontal_mask == swtich_class_2] = switch_class_1
        horizontal_mask[horizontal_mask == temp] = swtich_class_2

    if case == "case3":
        temp = 4
        switch_class_1 = 2
        swtich_class_2 = 3

        horizontal_mask[horizontal_mask == switch_class_1] = temp
        horizontal_mask[horizontal_mask == swtich_class_2] = switch_class_1
        horizontal_mask[horizontal_mask == temp] = swtich_class_2

    if case == "case4":
        temp = 4
        switch_class_1 = 2
        swtich_class_2 = 3

        horizontal_mask[horizontal_mask == switch_class_1] = temp
        horizontal_mask[horizontal_mask == swtich_class_2] = switch_class_1
        horizontal_mask[horizontal_mask == temp] = swtich_class_2

    if case == "case6":
        temp = 4
        switch_class_1 = 2
        swtich_class_2 = 3

        horizontal_mask[horizontal_mask == switch_class_1] = temp
        horizontal_mask[horizontal_mask == swtich_class_2] = switch_class_1
        horizontal_mask[horizontal_mask == temp] = swtich_class_2

    if case == "case7":
        temp = 4
        switch_class_1 = 2
        swtich_class_2 = 3

        horizontal_mask[horizontal_mask == switch_class_1] = temp
        horizontal_mask[horizontal_mask == swtich_class_2] = switch_class_1
        horizontal_mask[horizontal_mask == temp] = swtich_class_2

    if case == "case8":
        temp = 4
        switch_class_1 = 2
        swtich_class_2 = 3

        horizontal_mask[horizontal_mask == switch_class_1] = temp
        horizontal_mask[horizontal_mask == swtich_class_2] = switch_class_1
        horizontal_mask[horizontal_mask == temp] = swtich_class_2

    # 아래 video case는 vertical flip class change 해줘야 left, right curve가 딱 명확히 구분된 데이터 나옴
    if case == "case1":
        temp = 4
        switch_class_1 = 2
        swtich_class_2 = 3

        vertical_mask[vertical_mask == switch_class_1] = temp
        vertical_mask[vertical_mask == swtich_class_2] = switch_class_1
        vertical_mask[vertical_mask == temp] = swtich_class_2

    if case == "case5":
        temp = 4
        switch_class_1 = 2
        swtich_class_2 = 3

        vertical_mask[vertical_mask == switch_class_1] = temp
        vertical_mask[vertical_mask == swtich_class_2] = switch_class_1
        vertical_mask[vertical_mask == temp] = swtich_class_2




    return mask, horizontal_mask, vertical_mask


if __name__ == "__main__":
    base_path = os.path.dirname(os.path.abspath(__file__))
    train_data_path = os.path.join(base_path, "train_data")
    save_path = os.path.join(base_path, "train_flip_aug_data")
    save_image_path = os.path.join(base_path, "train_flip_aug_data", "image")
    save_mask_path = os.path.join(base_path, "train_flip_aug_data", "mask")

    if not os.path.exists(save_path):
        os.makedirs(save_path)

    if not os.path.exists(save_image_path):
        os.makedirs(save_image_path)

    if not os.path.exists(save_mask_path):
        os.makedirs(save_mask_path)

    image_path = os.path.join(train_data_path, "image")
    mask_path = os.path.join(train_data_path, "mask")

    image_list = os.listdir(image_path)
    mask_list = os.listdir(mask_path)

    sort_f = lambda f: int(''.join(filter(str.isdigit, f)))

    image_list = [file for file in image_list if file.endswith("png")]
    image_list.sort(key=sort_f)

    mask_list = [file for file in mask_list if file.endswith("png")]
    mask_list.sort(key=sort_f)

    horizontal_transform = A.Compose([
        A.HorizontalFlip(p=1)
    ])

    vertical_transform = A.Compose([
        A.VerticalFlip(p=1)
    ])

    image_count_num = 0
    mask_count_num = 0

    for img_name in image_list:

        image = cv2.imread(os.path.join(image_path, img_name), cv2.IMREAD_COLOR)
        mask = cv2.imread(os.path.join(mask_path, img_name), cv2.IMREAD_GRAYSCALE)

        horizon_transformed = horizontal_transform(image=image, mask=mask)
        vertical_transformed = vertical_transform(image=image, mask=mask)

        horizontal_image = horizon_transformed["image"]
        vertical_image = vertical_transformed["image"]

        horizontal_mask = horizon_transformed["mask"]
        vertical_mask = vertical_transformed["mask"]

        class_num = np.unique(mask).shape[0]
        if class_num > 2:
            mask, horizontal_mask, vertical_mask = mask_preprocessing(mask, horizontal_mask, vertical_mask, img_name)

        cv2.imwrite(os.path.join(save_image_path, "%04d.png"%image_count_num), image)
        image_count_num += 1
        cv2.imwrite(os.path.join(save_image_path, "%04d.png"%image_count_num), horizontal_image)
        image_count_num += 1
        cv2.imwrite(os.path.join(save_image_path, "%04d.png"%image_count_num), vertical_image)
        image_count_num += 1

        cv2.imwrite(os.path.join(save_mask_path, "%04d.png"%mask_count_num), mask)
        mask_count_num += 1
        cv2.imwrite(os.path.join(save_mask_path, "%04d.png"%mask_count_num), horizontal_mask)
        mask_count_num += 1
        cv2.imwrite(os.path.join(save_mask_path, "%04d.png"%mask_count_num), vertical_mask)
        mask_count_num += 1


