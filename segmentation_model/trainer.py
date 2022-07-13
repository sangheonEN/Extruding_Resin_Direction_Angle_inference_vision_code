import datetime
import os.path as osp
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
import torchvision.utils as vutils


class solver(object):
    def __init__(self, train_data_loader, valid_data_loader, opts, summary):
        self.data_loader_train = train_data_loader
        self.data_loader_valid = valid_data_loader
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
            self.optim, self.scheduler = optim_module.prepare_optim(opts, self.model)

        self.model.to(opts.cuda)



class Trainer(solver):
    def __init__(self, train_data_loader, valid_data_loader, opts, summary):
        super(Trainer, self).__init__(train_data_loader, valid_data_loader, opts, summary)
        self.cuda = opts.cuda
        self.opts = opts
        self.train_loader = train_data_loader
        self.val_loader = valid_data_loader
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
        self.max_iter = opts.cfg['max_iteration']
        self.best_mean_iu = 0

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

                # val loss function
                try:
                    if self.opts.loss_func == 'ce':
                        loss = loss_f.cross_entropy2d(score, target)
                        
                    elif self.opts.loss_func =='dice':
                        ce_loss = loss_f.cross_entropy2d(score, target)
                        dice = loss_f.DiceLoss(mode='multilabel', classes=[class_idx for class_idx in range(n_class)])
                        dice_loss = dice(score, target)
                        loss = dice_loss + ce_loss
                    
                    elif self.opts.loss_func == 'focal':
                        focal = loss_f.FocalLoss(alpha=0.25, gamma=2)
                        loss = focal(score, target)
                        
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
                np.squeeze(true)
                r = true.copy()
                g = true.copy()
                b = true.copy()
                for ll in range(0, n_class):
                    r[true == ll] = label_colours[ll][0]
                    g[true == ll] = label_colours[ll][1]
                    b[true == ll] = label_colours[ll][2]
                rgb = np.zeros((true.shape[0], true.shape[1], 3))
                rgb[:, :, 0] = b.squeeze()
                rgb[:, :, 1] = g.squeeze()
                rgb[:, :, 2] = r.squeeze()
                
                label_true_visual.append(rgb)
            
            for pred in label_preds:
                pred.squeeze()
                r = pred.copy()
                g = pred.copy()
                b = pred.copy()
                for ll in range(0, n_class):
                    r[pred == ll] = label_colours[ll][0]
                    g[pred == ll] = label_colours[ll][1]
                    b[pred == ll] = label_colours[ll][2]
                rgb = np.zeros((true.shape[0], true.shape[1], 3))
                rgb[:, :, 0] = b.squeeze()
                rgb[:, :, 1] = g.squeeze()
                rgb[:, :, 2] = r.squeeze()
                
                label_pred_visual.append(rgb)
            
            annotation_mask = torch.Tensor(np.array(label_true_visual)).permute(0, 3, 1, 2)
            prediction_mask = torch.Tensor(np.array(label_pred_visual)).permute(0, 3, 1, 2)
            annotation_mask_grid = vutils.make_grid(annotation_mask)
            prediction_mask_grid = vutils.make_grid(prediction_mask)
            self.summary.add_image(f'Valid_Anno_image_best_epoch:{self.epoch}', annotation_mask_grid, self.epoch)
            self.summary.add_image(f'Valid_Pred_image_best_epoch:{self.epoch}', prediction_mask_grid, self.epoch)
            
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
            self.model.train()

    def train_epoch(self):
        self.model.train()

        n_class = len(self.train_loader.dataset.class_names)

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
                    loss = loss_f.cross_entropy2d(score, target)
                elif self.opts.loss_func =='dice':
                    ce_loss = loss_f.cross_entropy2d(score, target)

                    dice = loss_f.DiceLoss(mode='multilabel', classes=[class_idx for class_idx in range(n_class)])
                    dice_loss = dice(score, target)
                    loss = dice_loss + ce_loss
                
                elif self.opts.loss_func == 'focal':
                    focal = loss_f.FocalLoss(mode='multilabel', alpha=0.5, gamma=2)
                    loss = focal(score, target)
                
                else:
                    print("there is not loss function")
            except Exception as e:
                print(e)
            
            loss /= len(data)
            if np.isnan(float(loss.item())):
                raise ValueError('loss is nan while training')

            self.optim.zero_grad()
            loss.backward()
            self.optim.step()
            self.scheduler.step()
            
            # Segmentation metrics calculation
            metrics = []
            lbl_pred = score.data.max(1)[1].cpu().numpy()[:, :, :]
            lbl_true = target.data.cpu().numpy()
            acc, acc_cls, mean_iu, fwavacc = \
                utils.label_accuracy_score(
                    lbl_true, lbl_pred, n_class=n_class)
            metrics.append((acc, acc_cls, mean_iu, fwavacc))
            metrics = np.mean(metrics, axis=0)
            
            # tensorboard train_loss, learning_rate, miou, acc
            self.summary.add_scalar('Train_loss', loss.item(), self.iteration)
            self.summary.add_scalar('learning_rate', self.optim.param_groups[0]['lr'], self.iteration)
            self.summary.add_scalar('Train_mIoU', mean_iu, self.iteration)
            self.summary.add_scalar('Train_Accuracy',acc, self.iteration)

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

    def Train(self):
        max_epoch = int(math.ceil(1. * self.max_iter / len(self.train_loader)))
        for epoch in tqdm.trange(self.epoch, max_epoch, desc='Train',
                                 ncols=80):
            self.epoch = epoch
            self.train_epoch()
            if self.iteration >= self.max_iter:
                break
