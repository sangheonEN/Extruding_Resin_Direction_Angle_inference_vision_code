import os
import torch
from torch.utils.data import DataLoader
from utils import get_config, get_log_dir, get_cuda
from inference_data_loader import *
from inferencer import Inference_f
import argparse
import warnings
warnings.filterwarnings('ignore')

# test_input_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'DATA', 'test_data', 'image')
# test_mask_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'DATA', 'test_data', 'mask')

test_input_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'DATA', 'IMAGE', 'case8')
test_mask_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'DATA', 'MASK', 'case8')

experiment_input_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'DATA', 'experiment_data', 'image')
experiment_mask_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'DATA', 'experiment_data', 'mask')

def data_sort_list(input_path, mask_path):
    sort_function = lambda f: int(''.join(filter(str.isdigit, f)))

    input_list = os.listdir(input_path)
    input_list = [file for file in input_list if file.endswith("png")]
    input_list.sort(key=sort_function)
    mask_list = os.listdir(mask_path)
    mask_list = [file for file in mask_list if file.endswith('png')]
    mask_list.sort(key=sort_function)

    return input_list, mask_list


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    # Model hyper-parameters
    parser.add_argument('--mode', type=str, default='inference', choices=['trainval', 'inference'])
    parser.add_argument("--gpu_id", type=int, default=0)
    parser.add_argument("--backbone", type=str, default='resnet')
    parser.add_argument("--model", type=str, default='deeplabv3', choices=['fcn', 'deeplabv3'])
    parser.add_argument("--resume", type=str, default='F:\KITECH\Die_Auto_Centering\code_work\segmentation_model\model_r2_front\logs\\best_model\model_best.pth.tar',
                        help='model saver path opts.out에서 log dir을 만들고 거기에 모델 결과 log와 ckpt 파일(the last and best model)이 저장된다'
                             'inference 상태일때 저장된 best model의 file path를 입력하면 best model을 load함.'
                             'train, val 상태일때 the last model의 file path를 입력하면 the last model을 load해서 연속적인 학습가능.')
    parser.add_argument("--backbone_layer", type=str, default='101', choices=['50', '101'])
    parser.add_argument("--loss_func", type=str, default='my_minwoo_focal', choices=['my_focal', 'my_minwoo_focal', 'ce', 'ce_effective', 'dice', 'focal', 'focal_dice',
                                                                                                 'focal_effective', 'focal_effective_square_sqrt'])
    opts = parser.parse_args()
    

    opts.cuda = get_cuda(torch.cuda.is_available() and opts.gpu_id != -1,
                         opts.gpu_id)
    print('Cuda', opts.cuda)
    cfg = get_config()[1]
    opts.cfg = cfg

    if opts.model == 'deeplabv3':
        if opts.mode in ['trainval', 'inference']:
            opts.out = get_log_dir('deeplabv3_' + opts.backbone_layer, cfg)
            print('Output logs: ', opts.out)
    else:
        if opts.mode in ['trainval', 'inference']:
            opts.out = get_log_dir('fcn_' + opts.backbone_layer, cfg)
            print('Output logs: ', opts.out)

    test_input_list, test_mask_list = data_sort_list(test_input_path, test_mask_path)

    test_data = CustomImageDataset(test_input_list, test_mask_list, test_input_path, test_mask_path,
                                   is_train_data=False)

    test_dataloader = DataLoader(test_data, batch_size=3, shuffle=False)

    experiment_input_list, experiment_mask_list = data_sort_list(experiment_input_path, experiment_mask_path)

    experiment_data = CustomImageDataset(experiment_input_list, experiment_mask_list, experiment_input_path, experiment_mask_path,
                                         is_train_data=False)

    experiment_dataloader = DataLoader(experiment_data, batch_size=3, shuffle=False)

    inference = Inference_f(test_dataloader, experiment_dataloader, opts)

    # inference Test or experiment
    inference.Test()
    # inference.experiment()

