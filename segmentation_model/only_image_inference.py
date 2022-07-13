import os
import torch
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from utils import get_config, get_log_dir, get_cuda
import argparse
import warnings
import numpy as np
from importlib import import_module
import cv2
warnings.filterwarnings('ignore')


def data_sort_list(input_path):
    sort_function = lambda f: int(''.join(filter(str.isdigit, f)))

    input_list = os.listdir(input_path)
    input_list = [file for file in input_list if file.endswith("png")]
    input_list.sort(key=sort_function)


    return input_list

LEE_COLORMAP = [
    [0, 0, 0],
    [255, 255, 255],
    [255, 0, 0],
    [0, 0, 255]
]



class solver_inf(object):
    def __init__(self, test_data_loader, opts):
        self.test_data_loader = test_data_loader
        self.num_class = len(self.test_data_loader.dataset.class_names)


        if opts.model == "deeplabv3":
            model_module = import_module('models.{}.deeplabv3_{}'.format(
                opts.backbone, opts.backbone_layer))
            self.model = model_module.Deeplabv3(n_class=self.num_class)
        else:
            model_module = import_module('models.{}.fcn_{}'.format(
                opts.backbone, opts.backbone_layer))
            self.model = model_module.FCN(n_class=self.num_class)

        self.model.resume(opts.resume, test=opts.mode in ['inference'])
        self.model.to(opts.cuda)

class Inference_f(solver_inf):
    def __init__(self, test_data_loader, opts):
        super(Inference_f, self).__init__(test_data_loader, opts)
        self.cuda = opts.cuda
        self.opts = opts
        self.test_data_loader = test_data_loader

        if opts.mode in ['inference']:
            return
        
    def mkdir_f(self, path):
    
        if not os.path.exists(path):
            os.makedirs(path)

            return
    
    def run_fromfile(self, model, img_file, cuda):
        img_torch = img_file
        img_torch = img_torch.to(cuda)
        model.eval()
        with torch.no_grad():
            score = model(img_torch)
        return score
    
    def save_image_and_pred(self, label_mask, img, n_classes, cnt):
        """Decode segmentation class labels into a color image

        Args:
            label_mask (np.ndarray): an (M,N) array of integer values denoting
            the class label at each spatial location.
            plot (bool, optional): whether to show the resulting color image
            in a figure.

        Returns:
            (np.ndarray, optional): the resulting decoded color image.
        """
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
       
        label_colours = LEE_COLORMAP
        r = label_mask.copy()
        g = label_mask.copy()
        b = label_mask.copy()
        for ll in range(0, n_classes):
            r[label_mask == ll] = label_colours[ll][0]
            g[label_mask == ll] = label_colours[ll][1]
            b[label_mask == ll] = label_colours[ll][2]
        rgb = np.zeros((label_mask.shape[0], label_mask.shape[1], 3))
        rgb[:, :, 0] = b.squeeze()
        rgb[:, :, 1] = g.squeeze()
        rgb[:, :, 2] = r.squeeze()
        
        self.mkdir_f(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'only_image_result'))

        self.mkdir_f(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'only_image_result', 'pred'))
        cv2.imwrite(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'only_image_result', 'pred', '%04d.png' % cnt), rgb)
                
        self.mkdir_f(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'only_image_result', 'origin_image'))
        cv2.imwrite(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'only_image_result', 'origin_image', '%04d.png' % cnt), img)

    def Test(self):
        count = 0
        label_preds = list()             

        for image in self.test_data_loader:
            lbl_pred = self.run_fromfile(self.model,
                                    image,
                                    self.opts.cuda)


            lbl_pred = lbl_pred.data.max(1)[1].cpu().numpy()[:, :, :]
            

            for img, lp in zip(image, lbl_pred):
                # 지금 torch에서 convolution operation을 위해 transform(data shape, normalization)을 한 상태니까
                # untransform을 통해 다시 shape과 normalization 변경해야함. 변경되는 내용은 메서드 내용에서 확인
                img = self.test_data_loader.dataset.untransform(img)
                
                # visualization
                self.save_image_and_pred(lp, img, self.num_class, cnt=count)
                
                label_preds.append(lp)
                count+=1


class CustomImageDataset(Dataset):
    
    def __init__(self, img_list, img_path, is_train_data=False):
        self.img_list = img_list
        self.img_path = img_path
        self.is_train_data = is_train_data

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, idx):
        image = cv2.imread(os.path.join(self.img_path, self.img_list[idx]))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        image = self.roi_crop(image)

        image = self.transform(image, self.is_train_data)
        
        return image

    def transform(self, img, _is_training):
        # train will transfrom in train_augmentation at utils.py

        if _is_training == False:
            mean_resnet = np.array([0.485, 0.456, 0.406])
            std_resnet = np.array([0.229, 0.224, 0.225])

            img = np.array(img, dtype=np.float64)
            img /= 255.
            img -= mean_resnet
            img /= std_resnet
            
            img = img.transpose(2, 0, 1)  # mask 는 h, w shape이니까 필요 없음??
            img = torch.from_numpy(img.copy()).float()
            
            return img
        
        else:
            
            img = img.transpose(2, 0, 1)  # mask 는 h, w shape이니까 필요 없음??
            lbl = lbl.transpose(2, 0, 1)  # mask 는 h, w shape이니까 필요 없음??
            img = torch.from_numpy(img.copy()).float()
            lbl = torch.from_numpy(lbl.copy()).long()
            
            return img, lbl

    def roi_crop(self, img):
        
        x, y, w, h = 400, 400, 540, 380
        
        img = img[y : y+h, x : x+w]
        
        return img


    class_names = np.array([
        'background',
        'object',
        'curve_1',
        'curve_2'
    ])

    def untransform(self, img):
        mean_resnet = np.array([0.485, 0.456, 0.406])
        std_resnet = np.array([0.229, 0.224, 0.225])

        # 입력은 channel, height, width형식으로 들어오니
        # height, width, channel shape으로 transpose 해야함.
        img = img.numpy()
        img = img.transpose(1, 2, 0)
        img *= std_resnet
        img += mean_resnet
        img *= 255
        img = img.astype(np.uint8)

        return img


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()

    # Model hyper-parameters
    parser.add_argument('--mode', type=str, default='inference', choices=['trainval', 'inference'])
    parser.add_argument("--gpu_id", type=int, default=0)
    parser.add_argument("--backbone", type=str, default='resnet')
    parser.add_argument("--model", type=str, default='deeplabv3', choices=['fcn', 'deeplabv3'])
    parser.add_argument("--resume", type=str, default='F:\KITECH\Die_Auto_Centering\code_work\segmentation_model\model_r2\experiments_result\\thrid\MODEL-deeplabv3_101\model_best.pth.tar',
                        help='model saver path opts.out에서 log dir을 만들고 거기에 모델 결과 log와 ckpt 파일(the last and best model)이 저장된다'
                             'inference 상태일때 저장된 best model의 file path를 입력하면 best model을 load함.'
                             'train, val 상태일때 the last model의 file path를 입력하면 the last model을 load해서 연속적인 학습가능.')
    parser.add_argument("--backbone_layer", type=str, default='101', choices=['50', '101'])
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
            
    
    
    
    test_input_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'DATA', 'only_image')

    test_input_list = data_sort_list(test_input_path)

    test_data = CustomImageDataset(test_input_list, test_input_path, is_train_data=False)

    test_dataloader = DataLoader(test_data, batch_size=3, shuffle=False)

    inference = Inference_f(test_dataloader, opts)

    inference.Test()
    
    
    
    

    

