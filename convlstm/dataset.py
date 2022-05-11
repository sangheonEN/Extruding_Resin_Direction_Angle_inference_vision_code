import numpy as np
import os
from torch.utils.data import DataLoader
import cv2
import torch


def data_loader(x, y, batch_size):

    x = DataLoader(x, batch_size=batch_size)
    y = DataLoader(y, batch_size=batch_size)

    return x, y


def list_sort(path):
    sort_function = lambda f: int(''.join(filter(str.isdigit, f)))

    data_list = os.listdir(path)
    data_list = [file for file in data_list if file.endswith('png')]
    data_list.sort(key=sort_function)

    return data_list

def data_set(img_path, img_list, flag):
    """
    :Description data set
    (final_num - start_num) / step = sequence

    - Center Point start image num: 428
    - Case 8 start image num: 326
    - Case 9 start image num: 319

    :return: input_data, target_data
    shape: N, sequence, channel, h, w

    """
    if flag == 'center_point':
        start_img_num = 428
    elif flag == 'case8':
        start_img_num = 326
    else:
        start_img_num = 319


    sequence = 22
    step = 30
    total_num = (step * sequence) + start_img_num
    total_data_list = list()
    total_y_data_list = list()
    x, y, w, h = 495, 501, 344, 267

    for data_num in range(0, 30):
        sequence_data_list = list()
        for idx in range(start_img_num+data_num-1, total_num+data_num-1, step):
            img = cv2.imread(os.path.join(img_path, img_list[idx]), cv2.IMREAD_COLOR)
            img_crop = img[y:y+h, x:x+w]
            img_resize = cv2.resize(img_crop, (384, 294), cv2.INTER_LINEAR)
            sequence_data_list.append(img_resize)

            # print(f"data_num:{data_num}, idx: {idx}, img_list[idx]: {img_list[idx]}")

            if len(sequence_data_list) > 22:
                continue

        # print(f"data_num: {data_num}, len: {len(sequence_data_list)}")

        final_data = np.array(sequence_data_list)
        total_data_list.append(final_data)
        if flag == 'center_point':
            total_y_data_list.append(torch.Tensor(0))
        elif flag == 'case8':
            total_y_data_list.append(torch.Tensor(1))
        else:
            total_y_data_list.append(torch.Tensor(2))

    return np.array(total_data_list).transpose(0, 1, 4, 2, 3), np.array(total_y_data_list)


# if __name__ == "__main__":
    # img_CP_path = './image_data/Case_CP_Top'
    # img_Case8_path = './image_data/Case_8_Top'
    # img_Case9_path = './image_data/Case_9_Top'
    # img_list_CP = os.listdir(img_CP_path)
    # img_list_8 = os.listdir(img_Case8_path)
    # img_list_9 = os.listdir(img_Case9_path)
    # img_list_CP = [file for file in img_list_CP if file.endswith('png')]
    # img_list_8 = [file for file in img_list_8 if file.endswith('png')]
    # img_list_9 = [file for file in img_list_9 if file.endswith('png')]
    # sort_function = lambda f: int(''.join(filter(str.isdigit, f)))
    # img_list_CP.sort(key= sort_function)
    # img_list_8.sort(key= sort_function)
    # img_list_9.sort(key= sort_function)
    #
    # x_cp, y_cp = data_set(img_CP_path, img_list_CP, flag='center_point')
    # x_8, y_8 = data_set(img_Case8_path, img_list_8, flag='case8')
    # x_9, y_9 = data_set(img_Case9_path, img_list_9, flag='case9')
    #
    # print(f"x_cp, y_cp shape: {x_cp.shape, y_cp.shape}")
    # print(f"x_8, y_8 shape: {x_8.shape, y_8.shape}")
    # print(f"x_9, y_9 shape: {x_9.shape, y_9.shape}")
    #
    # total_input_list = list()
    # total_target_list = list()
    #
    # total_input_list.append(x_cp)
    # total_input_list.append(x_8)
    # total_input_list.append(x_9)
    # total_target_list.append(y_cp)
    # total_target_list.append(y_8)
    # total_target_list.append(y_9)
    #
    # total_input_array = np.array(total_input_list).reshape(len(total_input_list)*x_cp.shape[0], x_cp.shape[1], x_cp.shape[2], x_cp.shape[3], x_cp.shape[4])
    # total_target_array = np.array(total_target_list).reshape(len(total_input_list)*y_cp.shape[0], y_cp.shape[1])
    #
    # x_train, x_test, y_train, y_test = train_test_split(total_input_array, total_target_array, test_size=0.1, shuffle=True, random_state=32)
    # x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.1, shuffle=True, random_state=32)
    #
    # print("zz")









