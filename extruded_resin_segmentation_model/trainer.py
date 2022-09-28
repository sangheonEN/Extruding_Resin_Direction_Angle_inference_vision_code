import datetime
import os.path as osp

import cv2
import torch
import loss_f
import numpy as np
import tqdm
import math
import pytz
import utils
import os
from importlib import import_module
import shutil
import torch.nn as nn
import torchvision.utils as vutils
from torchvision.utils import make_grid
from torch_ema import ExponentialMovingAverage


class solver(object):
    def __init__(self, train_data_loader, valid_data_loader, opts, summary, experiment_dataloader):
        self.data_loader_train = train_data_loader
        self.data_loader_valid = valid_data_loader
        self.experiment_dataloader = experiment_dataloader
        self.summary = summary

        num_class = len(self.data_loader_train.dataset.class_names)

        if opts.model == "deeplabv3":
            model_module = import_module('models.{}.deeplabv3_{}'.format(
                opts.backbone, opts.backbone_layer))
            self.model = model_module.Deeplabv3(n_class=num_class)
        else:
            model_module = import_module('models.{}.fcn_{}'.format(
                opts.backbone, opts.backbone_layer))
            self.model = model_module.FCN(n_class=num_class)

        self.model.resume(opts.resume, test=opts.mode in ['inference'])

        if opts.mode == 'trainval':
            optim_module = import_module('models.{}.helpers'.format(
                opts.backbone))
            self.optim, self.scheduler, self.ema = optim_module.prepare_optim(opts, self.model)

        self.model.to(opts.cuda)
        # self.ema.to(opts.cuda)



class Trainer(solver):
    def __init__(self, train_data_loader, valid_data_loader, opts, summary, experiment_dataloader):
        super(Trainer, self).__init__(train_data_loader, valid_data_loader, opts, summary, experiment_dataloader)
        self.cuda = opts.cuda
        self.opts = opts
        self.train_loader = train_data_loader
        self.val_loader = valid_data_loader
        self.experiment_dataloader = experiment_dataloader
        self.summary = summary


        if opts.mode in ['inference']:
            return

        self.timestamp_start = \
            datetime.datetime.now(pytz.timezone('Asia/Seoul'))

        self.interval_validate = opts.cfg.get('interval_validate',
                                              len(self.train_loader))
        if self.interval_validate is None:
            self.interval_validate = len(self.train_loader)

        self.out = opts.out
        if not osp.exists(self.out):
            os.makedirs(self.out)

        self.log_headers = [
            'epoch',
            'iteration',
            'train/loss',
            'train/acc',
            'train/acc_cls',
            'train/mean_iu',
            'train/fwavacc',
            'valid/loss',
            'valid/acc',
            'valid/acc_cls',
            'valid/mean_iu',
            'valid/fwavacc',
            'elapsed_time',
        ]
        if not osp.exists(osp.join(self.out, 'log.csv')):
            with open(osp.join(self.out, 'log.csv'), 'w') as f:
                f.write(','.join(self.log_headers) + '\n')

        self.epoch = 0
        self.iteration = 0
        self.val_iteration = 0
        self.best_conut = 0
        self.max_iter = opts.cfg['max_iteration']
        self.best_mean_iu = 0

    def experiment(self):
        training = self.model.training
        # model.eval() 을 호출하여 드롭아웃 및 배치 정규화를 평가 모드로 설정
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


                    elif self.opts.loss_func == 'LASD':
                        my_focal = loss_f.LASD(alpha=0.25, gamma=2, reduction='mean')
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
                    utils.decode_experiment_heatmap_save(img, sig, self.epoch, count)
                    count+=1


        if training:
            # experiment 끝난 후 다시 train mode로 변환
            self.model.train()

    def validate(self):
        training = self.model.training
        # model.eval() 을 호출하여 드롭아웃 및 배치 정규화를 평가 모드로 설정
        self.model.eval()

        n_class = len(self.val_loader.dataset.class_names)

        val_loss = 0
        label_trues, label_preds = [], []
        with torch.no_grad():

            # val data load
            for batch_idx, (data, target) in tqdm.tqdm(
                    enumerate(self.val_loader),
                    total=len(self.val_loader),
                    desc='Valid iteration=%d' % self.iteration,
                    ncols=80,
                    leave=False):
                data, target = data.to(self.cuda), target.to(self.cuda)
                score = self.model(data)

                # valid iteration
                val_iteration = batch_idx + self.epoch * len(self.val_loader)

                self.val_iteration = val_iteration

                # val loss function
                try:
                    if self.opts.loss_func == 'ce':
                        # loss = loss_f.cross_entropy2d(score, target)

                        cross_entropy = nn.CrossEntropyLoss(reduction= 'sum')
                        loss = cross_entropy(score, target.squeeze())

                    elif self.opts.loss_func == 'ce_effective':
                        cross_entropy = loss_f.cross_entropy_effective_number2d(score, target)
                        loss = cross_entropy

                    elif self.opts.loss_func == 'Hard_Easy_Loss':
                        my_focal = loss_f.Hard_Easy_Loss(alpha=0.25, gamma=2, reduction='sum', beta=2, threshold=0.3)
                        loss = my_focal(score, target)

                    elif self.opts.loss_func == 'LASD':
                        my_focal = loss_f.LASD(alpha=0.25, gamma=2, reduction='mean')
                        loss, sigma = my_focal(score, target)

                    elif self.opts.loss_func == 'focal':
                        focal = loss_f.FocalLoss(alpha=0.25, gamma=2, reduction='sum')
                        loss, sigma = focal(score, target)

                    elif self.opts.loss_func == 'focal_effective':
                        focal = loss_f.Focal_Effective_Loss(alpha=0.25, gamma=2, reduction='sum', balancing=True)
                        loss = focal(score, target)
                        # self.summary.add_scalar('class0 loss value', torch.mean(loss_flat[target_flat == 0]), self.val_iteration)
                        # self.summary.add_scalar('class1 loss value', torch.mean(loss_flat[target_flat == 1]), self.val_iteration)
                        # self.summary.add_scalar('class2 loss value', torch.mean(loss_flat[target_flat == 2]), self.val_iteration)
                        # self.summary.add_scalar('class3 loss value', torch.mean(loss_flat[target_flat == 3]), self.val_iteration)

                    elif self.opts.loss_func == 'focal_effective_square_sqrt':
                        focal = loss_f.Focal_Effective_Square_Sqrt_Loss(alpha=0.25, gamma=2, reduction='sum', balancing=True,
                                                                        flag_ss = 'sqrt', epoch= self.epoch)
                        loss = focal(score, target)
                        # self.summary.add_scalar('class0 loss value', torch.mean(loss_flat[target_flat == 0]), self.val_iteration)
                        # self.summary.add_scalar('class1 loss value', torch.mean(loss_flat[target_flat == 1]), self.val_iteration)
                        # self.summary.add_scalar('class2 loss value', torch.mean(loss_flat[target_flat == 2]), self.val_iteration)
                        # self.summary.add_scalar('class3 loss value', torch.mean(loss_flat[target_flat == 3]), self.val_iteration)

                    else:
                        print("there is not loss function")
                        
                except Exception as e:
                    print(e)


                if np.isnan(float(loss.item())):
                    raise ValueError('loss is nan while validating')
                val_loss += float(loss.item()) / len(data)

                # val metrics calculation
                imgs = data.data.cpu()
                lbl_pred = score.data.max(1)[1].cpu().numpy()[:, :, :]
                lbl_true = target.data.cpu()
                
                for img, lt, lp in zip(imgs, lbl_true, lbl_pred):
                    img, lt = self.val_loader.dataset.untransform(img, lt)
                    lp = np.expand_dims(lp, -1)
                    label_trues.append(lt)
                    label_preds.append(lp)
                

        metrics = utils.label_accuracy_score(label_trues, label_preds, n_class)

        val_loss /= len(self.val_loader)
        
        
        # tensorboard train_loss, learning_rate, miou, acc
        self.summary.add_scalar('Validation_loss', val_loss, self.epoch)
        self.summary.add_scalar('Val_mIoU', metrics[2], self.epoch)
        self.summary.add_scalar('Val_Accuracy',metrics[0], self.epoch)
        self.summary.add_scalar('Val_class_1_iou',metrics[4], self.epoch)
        self.summary.add_scalar('Val_class_2_iou',metrics[5], self.epoch)
        self.summary.add_scalar('Val_class_3_iou',metrics[6], self.epoch)
        self.summary.add_scalar('Val_class_4_iou',metrics[7], self.epoch)


        print(f"validation loss: {val_loss}\n")

        # val metric save
        with open(osp.join(self.out, 'log.csv'), 'a') as f:
            elapsed_time = (
                datetime.datetime.now(pytz.timezone('Asia/Seoul')) -
                self.timestamp_start).total_seconds()
            log = [self.epoch, self.iteration] + [''] * 5 + \
                  [val_loss] + list(metrics) + [elapsed_time]
            log = map(str, log)
            f.write(','.join(log) + '\n')

        # val best model save
        mean_iu = metrics[2]
        is_best = mean_iu > self.best_mean_iu
        if is_best:
            print(f"Update model best Epoch: {self.epoch}\n")
            self.best_mean_iu = mean_iu

            label_true_visual = list()
            label_pred_visual = list()

            label_colours = utils.LEE_COLORMAP

            for true in label_trues:
                class_num_true = np.unique(true).shape[0]

                np.squeeze(true)
                r = true.copy()
                g = true.copy()
                b = true.copy()
                for ll in range(0, class_num_true):
                    r[true == ll] = label_colours[ll][0]
                    g[true == ll] = label_colours[ll][1]
                    b[true == ll] = label_colours[ll][2]
                ture_rgb = np.zeros((true.shape[0], true.shape[1], 3))
                ture_rgb[:, :, 0] = b.squeeze()
                ture_rgb[:, :, 1] = g.squeeze()
                ture_rgb[:, :, 2] = r.squeeze()
                ture_rgb.astype(np.uint8)

                label_true_visual.append(ture_rgb)

            for pred in label_preds:
                class_num_pred = np.unique(pred).shape[0]

                pred.squeeze()
                r = pred.copy()
                g = pred.copy()
                b = pred.copy()
                for ll in range(0, class_num_pred):
                    r[pred == ll] = label_colours[ll][0]
                    g[pred == ll] = label_colours[ll][1]
                    b[pred == ll] = label_colours[ll][2]
                pred_rgb = np.zeros((true.shape[0], true.shape[1], 3))
                pred_rgb[:, :, 0] = b.squeeze()
                pred_rgb[:, :, 1] = g.squeeze()
                pred_rgb[:, :, 2] = r.squeeze()
                pred_rgb.astype(np.uint8)

                label_pred_visual.append(pred_rgb)

            val_pred_count = 0
            val_true_count = 0

            for label_true in label_true_visual:
                utils.decode_valid_segmap_save(label_true, self.epoch, flag_pred = False, cnt = val_pred_count)
                val_pred_count += 1

            for label_pred in label_pred_visual:
                utils.decode_valid_segmap_save(label_pred, self.epoch, flag_pred = True, cnt = val_true_count)
                val_true_count += 1


            # annotation_mask = torch.Tensor(np.array(label_true_visual, dtype=np.uint8)).permute(0, 3, 1, 2)
            # prediction_mask = torch.Tensor(np.array(label_pred_visual, dtype=np.uint8)).permute(0, 3, 1, 2)
            # # annotation_mask_grid = make_grid(annotation_mask)
            # # prediction_mask_grid = make_grid(prediction_mask)
            # self.summary.add_images(f'valid/Valid_Anno_images', annotation_mask, self.best_conut)
            # self.summary.add_images(f'valid/Valid_Pred_images', prediction_mask, self.best_conut)

            # self.best_conut += 1

        torch.save(
            {
                'epoch': self.epoch,
                'iteration': self.iteration,
                'arch': self.model.__class__.__name__,
                'optim_state_dict': self.optim.state_dict(),
                'model_state_dict': self.model.state_dict(),
                'best_mean_iu': self.best_mean_iu,
            }, osp.join(self.out, 'checkpoint.pth.tar'))
        if is_best:
            # best model name: model_best.pth.tar copy
            shutil.copy(osp.join(self.out, 'checkpoint.pth.tar'),
                        osp.join(self.out, 'model_best.pth.tar'))

        if training:
            # eval이 끝난 후 다시 train mode로 변환
            self.model.train()

    def train_epoch(self):
        self.model.train()

        n_class = len(self.train_loader.dataset.class_names)

        train_loss = 0

        # data load
        for batch_idx, (data, target) in tqdm.tqdm(
                enumerate(self.train_loader),
                total=len(self.train_loader),
                desc='Train epoch=%d' % self.epoch,
                ncols=80,
                leave=False):
            iteration = batch_idx + self.epoch * len(self.train_loader)
            if self.iteration != 0 and (iteration - 1) != self.iteration:
                continue  # for resuming
            self.iteration = iteration

            # validation start
            if self.iteration != 0 and self.iteration % self.interval_validate == 0:
                self.validate()

            assert self.model.training
            
            data, target = utils.augmentation_train(data, target)

            data, target = data.to(self.cuda), target.to(self.cuda)

            # optimization
            score = self.model(data)

            # loss function
            try:
                if self.opts.loss_func == 'ce':
                    # cross_entropy = loss_f.cross_entropy2d(score, target)
                    # loss = cross_entropy
                    cross_entropy = nn.CrossEntropyLoss(reduction='sum')
                    loss = cross_entropy(score, target.squeeze())

                elif self.opts.loss_func == 'ce_effective':
                    cross_entropy = loss_f.cross_entropy_effective_number2d(score, target)
                    loss = cross_entropy

                elif self.opts.loss_func == 'Hard_Easy_Loss':
                    my_focal = loss_f.Hard_Easy_Loss(alpha=0.25, gamma=2, reduction='sum', beta=2, threshold=0.3)
                    loss = my_focal(score, target)

                elif self.opts.loss_func == 'LASD':
                    my_focal = loss_f.LASD(alpha=0.25, gamma=2, reduction='mean')
                    loss, sigma = my_focal(score, target)

                elif self.opts.loss_func == 'focal':
                    focal = loss_f.FocalLoss(alpha=0.25, gamma=2, reduction='sum')
                    loss, sigma = focal(score, target)

                elif self.opts.loss_func == 'focal_effective':
                    focal = loss_f.Focal_Effective_Loss(alpha=0.25, gamma=2, reduction='sum', balancing=True)
                    loss = focal(score, target)

                elif self.opts.loss_func == 'focal_effective_square_sqrt':
                    focal = loss_f.Focal_Effective_Square_Sqrt_Loss(alpha=0.25, gamma=2, reduction='sum',
                                                                    balancing=True, flag_ss='sqrt', epoch=self.epoch)
                    loss = focal(score, target)

                else:
                    print("there is not loss function")
            except Exception as e:
                print(e)


            train_loss += float(loss.item()) / len(data)

            if np.isnan(float(loss.item())):
                raise ValueError('loss is nan while training')

            self.optim.zero_grad()
            loss.backward()
            self.optim.step()
            self.scheduler.step()
            # self.ema.update()
            
            # Segmentation metrics calculation
            metrics = []
            lbl_pred = score.data.max(1)[1].cpu().numpy()[:, :, :]
            lbl_true = target.data.cpu().numpy()
            acc, acc_cls, mean_iu, fwavacc, class_0_iou, class_1_iou, class_2_iou, class_3_iou = \
                utils.label_accuracy_score(
                    lbl_true, lbl_pred, n_class=n_class)
            metrics.append((acc, acc_cls, mean_iu, fwavacc))
            metrics = np.mean(metrics, axis=0)

            # tensorboard train_loss, learning_rate, miou, acc
            self.summary.add_scalar('Train_loss', loss.item(), self.iteration)
            self.summary.add_scalar('learning_rate', self.optim.param_groups[0]['lr'], self.iteration)
            self.summary.add_scalar('Train_mIoU', mean_iu, self.iteration)
            self.summary.add_scalar('Train_Accuracy', acc, self.iteration)

            # Segmentation metrics save
            with open(osp.join(self.out, 'log.csv'), 'a') as f:
                elapsed_time = (
                    datetime.datetime.now(pytz.timezone('Asia/Seoul')) -
                    self.timestamp_start).total_seconds()
                log = [self.epoch, self.iteration] + [loss.item()] + \
                    metrics.tolist() + [''] * 5 + [elapsed_time]
                log = map(str, log)
                f.write(','.join(log) + '\n')

            if self.iteration >= self.max_iter:
                break

        train_loss /= len(self.train_loader)

        # tensorboard train_loss, learning_rate, miou, acc
        self.summary.add_scalar('Train_loss_epoch', train_loss, self.epoch)


    def Train(self):
        max_epoch = int(math.ceil(1. * self.max_iter / len(self.train_loader)))
        for epoch in tqdm.trange(self.epoch, max_epoch, desc='Train',
                                 ncols=80):
            self.epoch = epoch
            self.train_epoch()
            if self.opts.loss_func == 'my_minwoo_focal':
                self.experiment()
            elif self.opts.loss_func == 'focal':
                self.experiment()
            else:
                pass
            if self.iteration >= self.max_iter:
                break
