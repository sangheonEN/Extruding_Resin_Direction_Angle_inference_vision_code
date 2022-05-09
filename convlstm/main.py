import torch
import argparse
import dataset
import trainer

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

parser = argparse.ArgumentParser()
parser.add_argument('-max_epoch', default=10, type=int)
parser.add_argument('-initial_lr', default=0.001, type=float)
parser.add_argument('-batch_size', default=10, type=int)
parser.add_argument('-frame_input_size', default=10, type=int)
parser.add_argument('-save_dir', default='./save_ckpt', type=str, help='save model parameters path')
parser.add_argument('-mode_flag', default='train', type=str, help='select the mode: [train], [inference]')
args = parser.parse_args()

if __name__ == '__main__':

    train_data, valid_data, test_data = dataset.data_set()


