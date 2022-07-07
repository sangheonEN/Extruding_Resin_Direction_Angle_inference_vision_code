import numpy as np
import os
import cv2
from sklearn.model_selection import train_test_split


def mkdir_f(path):

    if not os.path.exists(path):
        os.makedirs(path)


def mask_to_255():
    """
    matlab labeling mask data 0 or 1. all data multiply 255. and save
    """

    mask_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'MASK')



    file_list = os.listdir(mask_path)

    sort_f = lambda f: int(''.join(filter(str.isdigit, f)))


    for file in file_list:

        if file == 'all_data':
            continue

        img_list = os.listdir(os.path.join(mask_path, file))
        img_list = [file for file in img_list if file.endswith(".png")]
        img_list.sort(key=sort_f)

        for image in img_list:
            src = cv2.imread(os.path.join(mask_path, file, image), cv2.IMREAD_COLOR)

            src = src *255.

            cv2.imwrite(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'MASK', file, image), src)

            # cv2.imshow("zz", src)

            # cv2.waitKey(0)
            # cv2.destroyAllWindows()
            

def data_gather_one_dir():
    """
    Each case is splited into folders. So gather them in one folder and match the names of image and mask.  
    """

    mkdir_f(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'MASK', 'all_data'))
    mkdir_f(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'IMAGE', 'all_data'))

    
    mask_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'MASK')
    image_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'IMAGE')


    mask_file_list = os.listdir(mask_path)
    image_file_list = os.listdir(image_path)


    sort_f = lambda f: int(''.join(filter(str.isdigit, f)))

    count = 0
    for mask_file, image_file in zip(mask_file_list, image_file_list):

        if mask_file == 'all_data' or image_file == 'all_data':
            continue

        mask_list = os.listdir(os.path.join(mask_path, mask_file))
        mask_list = [file for file in mask_list if file.endswith(".png")]
        mask_list.sort(key=sort_f)

        img_list = os.listdir(os.path.join(image_path, image_file))
        img_list = [file for file in img_list if file.endswith(".png")]
        img_list.sort(key=sort_f)

        for mask, img in zip(mask_list, img_list):
            mask_ = cv2.imread(os.path.join(mask_path, mask_file, mask), cv2.IMREAD_COLOR)
            img_ = cv2.imread(os.path.join(image_path, image_file, img), cv2.IMREAD_COLOR)

            cv2.imwrite(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'IMAGE', 'all_data', '%04d.png'%count), img_)
            cv2.imwrite(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'MASK', 'all_data', '%04d.png'%count), mask_)

            count+=1


def train_valid_test_split_f():
    """
    split train, valid, test data
    """
    img_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "IMAGE", 'all_data')
    mask_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "MASK", 'all_data')

    train_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'train_data')
    valid_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'valid_data')
    test_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'test_data')

    train_path_img = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'train_data', 'image')
    train_path_mask = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'train_data', 'mask')
    valid_path_img = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'valid_data','image') 
    valid_path_mask = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'valid_data', 'mask') 
    test_path_img = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'test_data', 'image') 
    test_path_mask = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'test_data', 'mask') 

    mkdir_f(train_path)
    mkdir_f(valid_path)
    mkdir_f(test_path)
    mkdir_f(train_path_img)
    mkdir_f(train_path_mask)
    mkdir_f(valid_path_img)
    mkdir_f(valid_path_mask)
    mkdir_f(test_path_img)
    mkdir_f(test_path_mask)


    # 8 : 1 : 1 split 10 * 0.8 = 2, 8 * 0.25 = 2

    img_list = os.listdir(img_path)
    mask_list = os.listdir(mask_path)

    x_train, x_test, y_train, y_test = train_test_split(img_list, mask_list, test_size= 0.2, shuffle= True)
    
    x_train, x_valid, y_train, y_valid = train_test_split(x_train, y_train, test_size= 0.25, shuffle= True)

    if x_train == y_train and x_valid == y_valid and x_test == y_test:
        pass
    else:
        print("image and mask not match")
        raise
    
    for train_img_n, train_mask_n in zip(x_train, y_train):
        train_img = cv2.imread(os.path.join(img_path, train_img_n), cv2.IMREAD_COLOR)
        train_mask = cv2.imread(os.path.join(mask_path, train_mask_n), cv2.IMREAD_COLOR)

        cv2.imwrite(os.path.join(train_path_img, train_img_n), train_img)
        cv2.imwrite(os.path.join(train_path_mask, train_mask_n), train_mask)


    for valid_img_n, valid_mask_n in zip(x_valid, y_valid):
        valid_img = cv2.imread(os.path.join(img_path, valid_img_n), cv2.IMREAD_COLOR)
        valid_mask = cv2.imread(os.path.join(mask_path, valid_mask_n), cv2.IMREAD_COLOR)

        cv2.imwrite(os.path.join(valid_path_img, valid_img_n), valid_img)
        cv2.imwrite(os.path.join(valid_path_mask, valid_mask_n), valid_mask)


    for test_img_n, test_mask_n in zip(x_test, y_test):
        test_img = cv2.imread(os.path.join(img_path, test_img_n), cv2.IMREAD_COLOR)
        test_mask = cv2.imread(os.path.join(mask_path, test_mask_n), cv2.IMREAD_COLOR)

        cv2.imwrite(os.path.join(test_path_img, test_img_n), test_img)
        cv2.imwrite(os.path.join(test_path_mask, test_mask_n), test_mask)


if __name__ == '__main__':


    # data_gather_one_dir()
    # mask_to_255()
    train_valid_test_split_f()





