import argparse
from sklearn.model_selection import train_test_split
import torch

from dataset import *
import trainer

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

parser = argparse.ArgumentParser()
parser.add_argument('-max_epoch', default=10, type=int)
parser.add_argument('-initial_lr', default=0.001, type=float)
parser.add_argument('-batch_size', default=3, type=int)
parser.add_argument('-save_dir', default='./save_ckpt', type=str, help='save model parameters path')
parser.add_argument('-mode_flag', default='train', type=str, help='select the mode: [train], [inference]')
args = parser.parse_args()

if __name__ == '__main__':

    class_name = ['center_point', 'case8', 'case9']

    img_CP_path = '../image_data/Case_CP_Top'
    img_Case8_path = '../image_data/Case_8_Top'
    img_Case9_path = '../image_data/Case_9_Top'

    img_list_CP = list_sort(img_CP_path)
    img_list_8 = list_sort(img_Case8_path)
    img_list_9 = list_sort(img_Case9_path)

    x_cp, y_cp = data_set(img_CP_path, img_list_CP, flag=class_name[0])
    x_8, y_8 = data_set(img_Case8_path, img_list_8, flag=class_name[1])
    x_9, y_9 = data_set(img_Case9_path, img_list_9, flag=class_name[2])

    # print(f"x_cp, y_cp shape: {x_cp.shape, y_cp.shape}")
    # print(f"x_8, y_8 shape: {x_8.shape, y_8.shape}")
    # print(f"x_9, y_9 shape: {x_9.shape, y_9.shape}")

    total_input_list = list()
    total_target_list = list()

    total_input_list.append(x_cp)
    total_input_list.append(x_8)
    total_input_list.append(x_9)
    total_target_list.append(y_cp)
    total_target_list.append(y_8)
    total_target_list.append(y_9)

    total_input_array = np.array(total_input_list).reshape(len(total_input_list)*x_cp.shape[0], x_cp.shape[1], x_cp.shape[2], x_cp.shape[3])
    total_target_array = np.array(total_target_list).reshape(len(total_input_list)*y_cp.shape[0])

    x_train, x_test, y_train, y_test = train_test_split(total_input_array, total_target_array, test_size=0.1, shuffle=False, random_state=32)
    x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.1, shuffle=False, random_state=32)

    input_train, target_train = data_loader(x_train, y_train, args.batch_size)
    input_val, target_val = data_loader(x_val, y_val, args.batch_size)
    input_test, target_test = data_loader(x_test, y_test, args.batch_size)

    if args.mode_flag == 'train':
        trainer.train(input_train, target_train, input_val, target_val, args, device)
    else:
        trainer.inference(input_test, target_test, args, device)




































# import argparse
# from sklearn.model_selection import train_test_split
# import torch
# import trainer
# import dataset
# import dataset
# from torch.utils.data import DataLoader
#
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#
# parser = argparse.ArgumentParser()
# parser.add_argument('-max_epoch', default=10, type=int)
# parser.add_argument('-initial_lr', default=0.001, type=float)
# parser.add_argument('-batch_size', default=3, type=int)
# parser.add_argument('-save_dir', default='./save_ckpt', type=str, help='save model parameters path')
# parser.add_argument('-mode_flag', default='train', type=str, help='select the mode: [train], [inference]')
# args = parser.parse_args()
#
# def train_val_test_split(img_list_casecp, img_list_case8, img_list_case9):
#     total_list = list()
#     total_list.append(img_list_casecp)
#     total_list.append(img_list_case8)
#     total_list.append(img_list_case9)
#
#     return total_list
#
#
# if __name__ == '__main__':
#
#     # img_path[0]: caseCP, img_path[1]: case8, img_path[2]: case9
#     img_path = ['F:\KITECH\Die_Auto_Centering\model_src\src\image_data\Case_CP_Top',
#                    'F:\KITECH\Die_Auto_Centering\model_src\src\image_data\Case_8_Top',
#                    'F:\KITECH\Die_Auto_Centering\model_src\src\image_data\Case_9_Top']
#
#     total_img_list = list()
#
#     total_data_num = 660
#     start_image_num = 319
#     val_data_num = int(total_data_num * 0.1)
#     test_data_num = int(total_data_num * 0.1)
#     train_data_num = total_data_num - val_data_num - test_data_num
#
#     img_list_casecp = dataset.list_sort(img_path[0])
#     img_list_case8 = dataset.list_sort(img_path[1])
#     img_list_case9 = dataset.list_sort(img_path[2])
#
#     train_total_img_list = train_val_test_split(img_list_casecp[start_image_num : start_image_num+train_data_num],
#                                                 img_list_case8[start_image_num : start_image_num+train_data_num],
#                                                 img_list_case9[start_image_num : start_image_num+train_data_num])
#     val_total_img_list = train_val_test_split(img_list_casecp[start_image_num+train_data_num : start_image_num+train_data_num+val_data_num],
#                                               img_list_case8[start_image_num+train_data_num : start_image_num+train_data_num+val_data_num],
#                                               img_list_case9[start_image_num+train_data_num : start_image_num+train_data_num+val_data_num])
#     test_total_img_list = train_val_test_split(img_list_casecp[start_image_num+train_data_num+val_data_num : start_image_num+train_data_num+val_data_num+test_data_num],
#                                                img_list_case8[start_image_num+train_data_num+val_data_num : start_image_num+train_data_num+val_data_num+test_data_num],
#                                                img_list_case9[start_image_num+train_data_num+val_data_num : start_image_num+train_data_num+val_data_num+test_data_num])
#
#
#
#     train_dataset = dataset.CustomImageDataset(train_total_img_list, img_path)
#     val_dataset = dataset.CustomImageDataset(val_total_img_list, img_path)
#     test_dataset = dataset.CustomImageDataset(test_total_img_list, img_path)
#
#
#
#
#     # img_case8_data = dataset.CustomImageDataset(img_case8_list, img_Case8_path, flag='case_8', is_train=True)
#     # img_case9_data = dataset.CustomImageDataset(img_case9_list, img_Case9_path, flag='case_9', is_train=True)
#
#     # img_cp_dataloader = DataLoader(img_cp_data, batch_size=3, shuffle=True)
#     # img_case8_dataloader = DataLoader(img_case8_data, batch_size=3, shuffle=True)
#     # img_case9_dataloader = DataLoader(img_case9_data, batch_size=3, shuffle=True)
#
#     if args.mode_flag == 'train':
#         trainer.train()
#     else:
#         trainer.inference()
#
