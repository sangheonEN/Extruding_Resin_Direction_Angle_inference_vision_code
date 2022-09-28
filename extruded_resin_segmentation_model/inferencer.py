import datetime
import os.path as osp
import numpy as np
import pytz
import utils
import os
from importlib import import_module
import extract_angle


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
    def __init__(self, test_data_loader, experiment_dataloader, opts):
        super(Inference_f, self).__init__(test_data_loader, opts)
        self.cuda = opts.cuda
        self.opts = opts
        self.test_data_loader = test_data_loader
        self.experiment_dataloader = experiment_dataloader

        if opts.mode in ['inference']:
            return


    def Test(self):
        count = 0
        label_trues = list()
        label_preds = list()
        timestamp_start = datetime.datetime.now(pytz.timezone('Asia/Seoul'))

        log_headers = [
            'LIST_NUM',
            'TEST/acc',
            'TEST/acc_cls',
            'TEST/mean_iu',
            'TEST/fwavacc',
            'elapsed_time',
        ]
        
        if not osp.exists(osp.join(os.path.dirname(os.path.abspath(__file__)), 'inference_result')):
            os.makedirs(osp.join(os.path.dirname(os.path.abspath(__file__)), 'inference_result'))
            with open(osp.join(osp.join(os.path.dirname(os.path.abspath(__file__)), 'inference_result'), 'log_inference_result.csv'), 'w') as f:
                f.write(','.join(log_headers) + '\n')
                

        for image, label in self.test_data_loader:
            lbl_pred = utils.run_fromfile(self.model,
                                    image,
                                    self.opts.cuda)


            lbl_pred = lbl_pred.data.max(1)[1].cpu().numpy()[:, :, :]
            

            for img, lt, lp in zip(image, label, lbl_pred):
                # 지금 torch에서 convolution operation을 위해 transform(data shape, normalization)을 한 상태니까
                # untransform을 통해 다시 shape과 normalization 변경해야함. 변경되는 내용은 메서드 내용에서 확인
                img, lt = self.test_data_loader.dataset.untransform(img, lt)
                lp = np.expand_dims(lp, -1)
                
                # visualization
                utils.decode_segmap_save(lp.squeeze(), self.num_class, flag_pred=True, cnt=count)
                utils.decode_segmap_save(lt, self.num_class, flag_pred=False, cnt=count)

                label_trues.append(lt)
                label_preds.append(lp)
                count+=1


        metrics = utils.label_accuracy_score(label_trues, label_preds, n_class=self.num_class)

        with open(osp.join(os.path.dirname(os.path.abspath(__file__)), 'inference_result', 'log_inference_result.csv'), 'a') as f:
            elapsed_time = (
                    datetime.datetime.now(pytz.timezone('America/Bogota')) -
                    timestamp_start).total_seconds()
            log = [count] + list(metrics) + [elapsed_time]
            log = map(str, log)
            f.write(','.join(log) + '\n')

    def experiment(self):
        import torch
        import torch.nn as nn
        import loss_f
        import tqdm
        import cv2
        # model.eval() 을 호출하여 드롭아웃 및 배치 정규화를 평가 모드로 설정

        count = 0

        for image, label in self.experiment_dataloader:
            lbl_pred = utils.run_fromfile(self.model,
                                          image,
                                          self.opts.cuda)

            lbl_pred = lbl_pred.data.max(1)[1].cpu().numpy()[:, :, :]

            for img, lt, lp in zip(image, label, lbl_pred):
                # 지금 torch에서 convolution operation을 위해 transform(data shape, normalization)을 한 상태니까
                # untransform을 통해 다시 shape과 normalization 변경해야함. 변경되는 내용은 메서드 내용에서 확인

                img, lt = self.experiment_dataloader.dataset.untransform(img, lt)

                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

                lp = np.expand_dims(lp, -1)

                # visualization
                utils.decode_segmap_save(lp.squeeze(), self.num_class, flag_pred=True, cnt=count)
                utils.decode_segmap_save(lt, self.num_class, flag_pred=False, cnt=count)

                utils.mkdir_f(os.path.join(os.path.dirname(os.path.abspath(
                    __file__)), 'inference_result', 'inference_input'))
                cv2.imwrite(os.path.join(os.path.dirname(os.path.abspath(
                    __file__)), 'inference_result', 'inference_input', '%04d.png' % count), img)

                count += 1

        self.model.eval()

        with torch.no_grad():

            # val data load
            for batch_idx, (data, target) in tqdm.tqdm(
                    enumerate(self.experiment_dataloader),
                    total=len(self.experiment_dataloader),
                    leave=False):
                data, target = data.to(self.cuda), target.to(self.cuda)
                score = self.model(data)

                # val loss function
                try:
                    if self.opts.loss_func == 'ce':
                        # loss = loss_f.cross_entropy2d(score, target)

                        cross_entropy = nn.CrossEntropyLoss(reduction='sum')
                        loss = cross_entropy(score, target.squeeze())

                    elif self.opts.loss_func == 'focal':
                        focal = loss_f.FocalLoss(alpha=0.25, gamma=2, reduction='sum')
                        loss, sigma = focal(score, target)


                    elif self.opts.loss_func == 'my_minwoo_focal':
                        my_focal = loss_f.My_Minwoo_FocalLoss(alpha=0.25, gamma=2, reduction='sum')
                        loss, sigma = my_focal(score, target)

                    else:
                        print("there is not loss function")

                except Exception as e:
                    print(e)

                # sigB, H, W -> H, W, img B, C, H, W -> H, W, C
                count = 0
                for img, sig in zip(data, sigma):

                    # img H, W, C. sig H, W
                    img = utils.untransform(img)
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    # sig = sig.unsqueeze(dim=-1)

                    # save file
                    utils.decode_experiment_heatmap_save(img, sig, 0, count)
                    count+=1
