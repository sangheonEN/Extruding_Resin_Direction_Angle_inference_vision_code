from torch.utils.data import Dataset
from torch.utils.data import ConcatDataset
import numpy as np
import cv2
import os
import albumentations as A
import torch



class_name = np.array([
    'case_8',
    'case_9',
    'case_cp'
])


def list_sort(data_path):
    sort_f = lambda f: int(''.join(filter(str.isdigit, f)))

    data_list = os.listdir(data_path)
    data_list = [file for file in data_list if file.endswith("png")]
    data_list.sort(key=sort_f)

    return data_list


class CustomImageDataset(Dataset):
    def __init__(self, img_list, img_path, class_num):

        self.img_list = img_list
        self.img_path = img_path
        self.class_num = class_num


    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, idx):

        img = cv2.imread(os.path.join(self.img_path, self.img_list[idx]), cv2.IMREAD_COLOR)

        img = self.transform(img)

        if self.class_num == 0:

            target = torch.tensor(0)
        
        elif self.class_num == 1:

            target = torch.tensor(1)
        
        else:
            target = torch.tensor(2)

        return img, target

    def transform(self, img):

        x, y, w, h = 495, 501, 344, 267

        img_crop = img[y:y + h, x:x + w]

        mean_resnet = np.array([0.485, 0.456, 0.406])
        std_resnet = np.array([0.229, 0.224, 0.225])

        img_crop = np.array(img_crop, dtype=np.float32)
        img_crop /= 255.
        img_crop -= mean_resnet
        img_crop /= std_resnet

        transform = A.Compose([
            A.Resize(384, 294)
        ])

        transformed = transform(image=img_crop)

        img_crop = transformed['image']

        img_crop = img_crop.transpose(2, 0, 1)
        img_crop = torch.from_numpy(img_crop.copy()).float()

        return img_crop


    def list_sort(self, data_path):
        sort_f = lambda f: int(''.join(filter(str.isdigit, f)))

        data_list = os.listdir(data_path)
        data_list = [file for file in data_list if file.endswith("png")]
        data_list.sort(key=sort_f)

        return data_list


if __name__ == '__main__':

    img_path_case_9 = './data/case_9'
    img_path_case_8 = './data/case_8'
    img_path_case_cp = './data/case_cp'
    img_list = os.listdir(img_path)
    img_list = [file for file in img_list if file.endswith('png')]

    input_data_case_9 = CustomImageDataset(img_list, img_path_case_9, class_num=0)
    input_data_case_8 = CustomImageDataset(img_list, img_path_case_8, class_num=2)
    input_data_case_cp = CustomImageDataset(img_list, img_path_case_cp, class_num=1)

    

    

    






    print("zz")
    
    

