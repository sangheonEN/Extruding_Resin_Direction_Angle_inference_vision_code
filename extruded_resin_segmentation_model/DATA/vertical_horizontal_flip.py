import cv2
import os
import albumentations as A
import numpy as np

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


def mask_flip_four_class_save(mask, horizontal_mask, vertical_mask, mask_name, case, mask_vertical_save_path,
                          mask_horizontal_save_path,
                          mask_origin_mask_save_path):


    if not os.path.exists(os.path.join(mask_origin_mask_save_path, case)):
        os.makedirs(os.path.join(mask_origin_mask_save_path, case))

    if not os.path.exists(os.path.join(mask_horizontal_save_path, case)):
        os.makedirs(os.path.join(mask_horizontal_save_path, case))

    if not os.path.exists(os.path.join(mask_vertical_save_path, case)):
        os.makedirs(os.path.join(mask_vertical_save_path, case))

    cv2.imwrite(os.path.join(mask_origin_mask_save_path, case, f"{mask_name}"), mask)
    cv2.imwrite(os.path.join(mask_horizontal_save_path, case, f"{mask_name}"), horizontal_mask)
    cv2.imwrite(os.path.join(mask_vertical_save_path, case, f"{mask_name}"), vertical_mask)



def mask_flip_save(mask, mask_name, case, mask_vertical_save_path, mask_horizontal_save_path, mask_origin_mask_save_path):


    if not os.path.exists(os.path.join(mask_origin_mask_save_path, case)):
        os.makedirs(os.path.join(mask_origin_mask_save_path, case))

    if not os.path.exists(os.path.join(mask_horizontal_save_path, case)):
        os.makedirs(os.path.join(mask_horizontal_save_path, case))

    if not os.path.exists(os.path.join(mask_vertical_save_path, case)):
        os.makedirs(os.path.join(mask_vertical_save_path, case))

    horizon_transformed = horizontal_transform(image=mask)
    vertical_transformed = vertical_transform(image=mask)

    horizontal_mask = horizon_transformed["image"]
    vertical_mask = vertical_transformed["image"]

    cv2.imwrite(os.path.join(mask_origin_mask_save_path, case, f"{mask_name}"), mask)
    cv2.imwrite(os.path.join(mask_horizontal_save_path, case, f"{mask_name}"), horizontal_mask)
    cv2.imwrite(os.path.join(mask_vertical_save_path, case, f"{mask_name}"), vertical_mask)



def image_flip_save(img_list, image_path, case, origin_mask_save_path, horizontal_save_path, vertical_save_path):
    for img_name in img_list:

        image = cv2.imread(os.path.join(image_path, case, img_name), cv2.IMREAD_COLOR)

        if not os.path.exists(os.path.join(origin_mask_save_path, case)):
            os.makedirs(os.path.join(origin_mask_save_path, case))

        if not os.path.exists(os.path.join(horizontal_save_path, case)):
            os.makedirs(os.path.join(horizontal_save_path, case))

        if not os.path.exists(os.path.join(vertical_save_path, case)):
            os.makedirs(os.path.join(vertical_save_path, case))

        horizon_transformed = horizontal_transform(image=image)
        vertical_transformed = vertical_transform(image=image)

        horizontal_image = horizon_transformed["image"]
        vertical_image = vertical_transformed["image"]

        cv2.imwrite(os.path.join(origin_mask_save_path, case, f"{img_name}"), image)
        cv2.imwrite(os.path.join(horizontal_save_path, case, f"{img_name}"), horizontal_image)
        cv2.imwrite(os.path.join(vertical_save_path, case, f"{img_name}"), vertical_image)



if __name__ == "__main__":

    base_path = os.path.dirname(os.path.abspath(__file__))
    mask_path = os.path.join(base_path, "MASK")
    image_path = os.path.join(base_path, "IMAGE")

    save_path = os.path.join(base_path, "mask_vertical_horizontal_visualization")
    mask_save_path = os.path.join(base_path, "mask_vertical_horizontal_mask")
    image_save_path = os.path.join(base_path, "image_vertical_horizontal_mask")

    if not os.path.exists(save_path):
        os.makedirs(save_path)
        os.makedirs(os.path.join(save_path, "vertical"))
        os.makedirs(os.path.join(save_path, "horizontal"))
        os.makedirs(os.path.join(save_path, "origin"))

        os.makedirs(mask_save_path)
        os.makedirs(os.path.join(mask_save_path, "vertical"))
        os.makedirs(os.path.join(mask_save_path, "horizontal"))
        os.makedirs(os.path.join(mask_save_path, "origin"))

        os.makedirs(image_save_path)
        os.makedirs(os.path.join(image_save_path, "vertical"))
        os.makedirs(os.path.join(image_save_path, "horizontal"))
        os.makedirs(os.path.join(image_save_path, "origin"))


    vertical_save_path = os.path.join(save_path, "vertical")
    horizontal_save_path = os.path.join(save_path, "horizontal")
    origin_mask_save_path = os.path.join(save_path, "origin")

    mask_vertical_save_path = os.path.join(mask_save_path, "vertical")
    mask_horizontal_save_path = os.path.join(mask_save_path, "horizontal")
    mask_origin_mask_save_path = os.path.join(mask_save_path, "origin")

    image_vertical_save_path = os.path.join(image_save_path, "vertical")
    image_horizontal_save_path = os.path.join(image_save_path, "horizontal")
    image_origin_mask_save_path = os.path.join(image_save_path, "origin")

    case_list = os.listdir(mask_path)

    sort_f = lambda f: int(''.join(filter(str.isdigit, f)))

    label_colors = LEE_COLORMAP
    label_colors_two_class = LEE_COLORMAP_two_class

    horizontal_transform = A.Compose([
        A.HorizontalFlip(p=1)
    ])

    vertical_transform = A.Compose([
        A.VerticalFlip(p=1)
    ])

    for case in case_list:
        if case == "all_data":
            continue

        mask_list = os.listdir(os.path.join(mask_path, case))
        mask_list = [file for file in mask_list if file.endswith("png")]
        mask_list.sort(key=sort_f)

        image_list = os.listdir(os.path.join(image_path, case))

        image_flip_save(image_list, image_path, case, image_origin_mask_save_path, image_horizontal_save_path, image_vertical_save_path)

        for mask_name in mask_list:

            mask = cv2.imread(os.path.join(mask_path, case, mask_name), cv2.IMREAD_GRAYSCALE)

            class_num = np.unique(mask).shape[0]

            if class_num < 3:

                mask_flip_save(mask, mask_name, case, mask_vertical_save_path, mask_horizontal_save_path, mask_origin_mask_save_path)

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

                if not os.path.exists(os.path.join(origin_mask_save_path, case)):
                    os.makedirs(os.path.join(origin_mask_save_path, case))

                if not os.path.exists(os.path.join(horizontal_save_path, case)):
                    os.makedirs(os.path.join(horizontal_save_path, case))

                if not os.path.exists(os.path.join(vertical_save_path, case)):
                    os.makedirs(os.path.join(vertical_save_path, case))

                horizon_transformed = horizontal_transform(image=rgb)
                vertical_transformed = vertical_transform(image=rgb)

                horizontal_mask = horizon_transformed["image"]
                vertical_mask = vertical_transformed["image"]


                cv2.imwrite(os.path.join(origin_mask_save_path, case, f"{mask_name}"), rgb)
                cv2.imwrite(os.path.join(horizontal_save_path, case, f"{mask_name}"), horizontal_mask)
                cv2.imwrite(os.path.join(vertical_save_path, case, f"{mask_name}"), vertical_mask)


                continue


            # class > 3 일 경우 = curve 외곽선 있는 데이터일 경우에는 flip 먼저하고 나중에 색칠
            horizon_transformed = horizontal_transform(image=mask)
            vertical_transformed = vertical_transform(image=mask)

            horizontal_mask = horizon_transformed["image"]
            horizontal_mask = np.expand_dims(horizontal_mask, -1)
            vertical_mask = vertical_transformed["image"]
            vertical_mask = np.expand_dims(vertical_mask, -1)


            # 아래 video case는 origin change 해줘야 left, right curve가 딱 명확히 구분된 데이터 나옴
            if case == "case0":
                temp = 4
                switch_class_1 = 2
                swtich_class_2 = 3

                mask[mask == switch_class_1] = temp
                mask[mask == swtich_class_2] = switch_class_1
                mask[mask == temp] = swtich_class_2

            if case == "case2":
                temp = 4
                switch_class_1 = 2
                swtich_class_2 = 3

                mask[mask == switch_class_1] = temp
                mask[mask == swtich_class_2] = switch_class_1
                mask[mask == temp] = swtich_class_2

            if case == "case3":
                temp = 4
                switch_class_1 = 2
                swtich_class_2 = 3

                mask[mask == switch_class_1] = temp
                mask[mask == swtich_class_2] = switch_class_1
                mask[mask == temp] = swtich_class_2

            if case == "case4":
                temp = 4
                switch_class_1 = 2
                swtich_class_2 = 3

                mask[mask == switch_class_1] = temp
                mask[mask == swtich_class_2] = switch_class_1
                mask[mask == temp] = swtich_class_2

            if case == "case5":
                temp = 4
                switch_class_1 = 2
                swtich_class_2 = 3

                mask[mask == switch_class_1] = temp
                mask[mask == swtich_class_2] = switch_class_1
                mask[mask == temp] = swtich_class_2



            # # 아래 video case는 horizontal flip class change 해줘야 left, right curve가 딱 명확히 구분된 데이터 나옴
            # if case == "case0":
            #     temp = 4
            #     switch_class_1 = 2
            #     swtich_class_2 = 3
            #
            #     horizontal_mask[horizontal_mask == switch_class_1] = temp
            #     horizontal_mask[horizontal_mask == swtich_class_2] = switch_class_1
            #     horizontal_mask[horizontal_mask == temp] = swtich_class_2
            #
            # if case == "case2":
            #     temp = 4
            #     switch_class_1 = 2
            #     swtich_class_2 = 3
            #
            #     horizontal_mask[horizontal_mask == switch_class_1] = temp
            #     horizontal_mask[horizontal_mask == swtich_class_2] = switch_class_1
            #     horizontal_mask[horizontal_mask == temp] = swtich_class_2
            #
            # if case == "case3":
            #     temp = 4
            #     switch_class_1 = 2
            #     swtich_class_2 = 3
            #
            #     horizontal_mask[horizontal_mask == switch_class_1] = temp
            #     horizontal_mask[horizontal_mask == swtich_class_2] = switch_class_1
            #     horizontal_mask[horizontal_mask == temp] = swtich_class_2
            #
            # if case == "case4":
            #     temp = 4
            #     switch_class_1 = 2
            #     swtich_class_2 = 3
            #
            #     horizontal_mask[horizontal_mask == switch_class_1] = temp
            #     horizontal_mask[horizontal_mask == swtich_class_2] = switch_class_1
            #     horizontal_mask[horizontal_mask == temp] = swtich_class_2
            #

            if case == "case1":
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

            if case == "case6":
                temp = 4
                switch_class_1 = 2
                swtich_class_2 = 3

                vertical_mask[vertical_mask == switch_class_1] = temp
                vertical_mask[vertical_mask == swtich_class_2] = switch_class_1
                vertical_mask[vertical_mask == temp] = swtich_class_2

            if case == "case7":
                temp = 4
                switch_class_1 = 2
                swtich_class_2 = 3

                vertical_mask[vertical_mask == switch_class_1] = temp
                vertical_mask[vertical_mask == swtich_class_2] = switch_class_1
                vertical_mask[vertical_mask == temp] = swtich_class_2

            if case == "case8":
                temp = 4
                switch_class_1 = 2
                swtich_class_2 = 3

                vertical_mask[vertical_mask == switch_class_1] = temp
                vertical_mask[vertical_mask == swtich_class_2] = switch_class_1
                vertical_mask[vertical_mask == temp] = swtich_class_2



            mask_flip_four_class_save(mask, horizontal_mask, vertical_mask, mask_name, case, mask_vertical_save_path, mask_horizontal_save_path,
                           mask_origin_mask_save_path)


            # visualization
            r = mask.copy()
            g = mask.copy()
            b = mask.copy()

            h_r = horizontal_mask.copy()
            h_g = horizontal_mask.copy()
            h_b = horizontal_mask.copy()

            v_r = vertical_mask.copy()
            v_g = vertical_mask.copy()
            v_b = vertical_mask.copy()

            for ll in range(0, 4):
                r[mask==ll] = label_colors[ll][0]
                g[mask==ll] = label_colors[ll][1]
                b[mask==ll] = label_colors[ll][2]

                h_r[horizontal_mask==ll] = label_colors[ll][0]
                h_g[horizontal_mask==ll] = label_colors[ll][1]
                h_b[horizontal_mask==ll] = label_colors[ll][2]

                v_r[vertical_mask==ll] = label_colors[ll][0]
                v_g[vertical_mask==ll] = label_colors[ll][1]
                v_b[vertical_mask==ll] = label_colors[ll][2]

            rgb = np.zeros((mask.shape[0], mask.shape[1], 3))
            h_rgb = np.zeros((horizontal_mask.shape[0], horizontal_mask.shape[1], 3))
            v_rgb = np.zeros((vertical_mask.shape[0], vertical_mask.shape[1], 3))

            rgb[:, :, 0] = b.squeeze()
            rgb[:, :, 1] = g.squeeze()
            rgb[:, :, 2] = r.squeeze()

            h_rgb[:, :, 0] = h_b.squeeze()
            h_rgb[:, :, 1] = h_g.squeeze()
            h_rgb[:, :, 2] = h_r.squeeze()

            v_rgb[:, :, 0] = v_b.squeeze()
            v_rgb[:, :, 1] = v_g.squeeze()
            v_rgb[:, :, 2] = v_r.squeeze()


            if not os.path.exists(os.path.join(origin_mask_save_path, case)):
                os.makedirs(os.path.join(origin_mask_save_path, case))

            if not os.path.exists(os.path.join(horizontal_save_path, case)):
                os.makedirs(os.path.join(horizontal_save_path, case))

            if not os.path.exists(os.path.join(vertical_save_path, case)):
                os.makedirs(os.path.join(vertical_save_path, case))

            cv2.imwrite(os.path.join(origin_mask_save_path, case, f"{mask_name}"), rgb)
            cv2.imwrite(os.path.join(horizontal_save_path, case, f"{mask_name}"), h_rgb)
            cv2.imwrite(os.path.join(vertical_save_path, case, f"{mask_name}"), v_rgb)








