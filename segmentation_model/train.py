import os
import torch
from torch.utils.data import DataLoader
from utils import get_config, get_log_dir, get_cuda, data_sort_list
from data_loader import *
from trainer import Trainer
import argparse
import warnings
from tensorboardX import SummaryWriter
warnings.filterwarnings('ignore')

train_input_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'DATA', 'train_data', 'image')
train_mask_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'DATA', 'train_data', 'mask')
validation_input_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'DATA', 'valid_data', 'image')
validation_mask_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'DATA', 'valid_data', 'mask')


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    # Model hyper-parameters
    parser.add_argument('--mode', type=str, default='trainval', choices=['trainval', 'inference'])
    parser.add_argument("--gpu_id", type=int, default=0)
    parser.add_argument("--backbone", type=str, default='resnet')
    parser.add_argument("--model", type=str, default='deeplabv3', choices=['fcn', 'deeplabv3'])
    parser.add_argument("--resume", type=str, default='',
                        help='model saver path opts.out에서 log dir을 만들고 거기에 모델 결과 log와 ckpt 파일(the last and best model)이 저장된다'
                             'inference 상태일때 저장된 best model의 file path를 입력하면 best model을 load함.'
                             'train, val 상태일때 the last model의 file path를 입력하면 the last model을 load해서 연속적인 학습가능.')
    parser.add_argument("--backbone_layer", type=str, default='101', choices=['50', '101'])
    parser.add_argument("--optim", type=str, default='adam', choices=['adam', 'sgd'])
    parser.add_argument("--lr_scheduler", type=str, default='CosineAnnealingWarmRestarts', choices=['steplr', 'CosineAnnealingWarmRestarts', 'LambdaLR'])
    parser.add_argument("--loss_func", type=str, default='focal', choices=['ce', 'dice', 'focal'])

    opts = parser.parse_args()

    # os.environ['CUDA_VISIBLE_DEVICES'] = str(opts.gpu_id)
    opts.cuda = get_cuda(torch.cuda.is_available() and opts.gpu_id != -1,
                         opts.gpu_id)
    print('Cuda', opts.cuda)
    cfg = get_config()[1]
    opts.cfg = cfg

    if opts.model == 'deeplabv3':
        if opts.mode in ['train', 'trainval']:
            opts.out = get_log_dir('deeplabv3_' + opts.backbone_layer, cfg)
            print('Output logs: ', opts.out)
    else:
        if opts.mode in ['train', 'trainval']:
            opts.out = get_log_dir('fcn_' + opts.backbone_layer, cfg)
            print('Output logs: ', opts.out)
            
    summary = SummaryWriter()
    
    train_input_list, train_mask_list = data_sort_list(train_input_path, train_mask_path)
    val_input_list, val_mask_list = data_sort_list(validation_input_path, validation_mask_path)

    training_data = CustomImageDataset(train_input_list, train_mask_list, train_input_path, train_mask_path,
                                       is_train_data=True)
    validation_data = CustomImageDataset(val_input_list, val_mask_list, validation_input_path, validation_mask_path,
                                         is_train_data=False)

    train_dataloader = DataLoader(training_data, batch_size=3, shuffle=True)
    valid_dataloader = DataLoader(validation_data, batch_size=3, shuffle=False)

    trainer = Trainer(train_dataloader, valid_dataloader, opts, summary)

    trainer.Train()









