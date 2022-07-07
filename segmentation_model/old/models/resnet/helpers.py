import torch
from torch.optim.lr_scheduler import *

#
# class Learning_rate_scheduler:
#     def __init__(self, optim, opts):
#         self.optim = optim
#
#     def stepLR(self):
#
#         scheduler = StepLR(self.optim, step_size=)
#



def prepare_optim(opts, model):

    if opts.optim == "adam":
        optim = torch.optim.Adam(model.parameters(),
                                 lr=opts.cfg['lr'],
                                 weight_decay=opts.cfg['weight_decay'])
    elif opts.optim == 'sgd':
        optim = torch.optim.SGD(model.paramters(),
                                lr=opts.cfg['lr'],
                                momentum=opts.cfg['momentum'],
                                weight_decay=opts.cfg['weight_decay']
                                )
    else:
        print("Could not input argment optimizer.")
        raise

    # 저장된 모델이 있으면 파라미터 불러옴.
    if opts.resume:
        checkpoint = torch.load(opts.resume)
        optim.load_state_dict(checkpoint['optim_state_dict'])



    # learning_rate scheduler
    # 코드 작성 필요.
    # lr_schedule = Learning_rate_scheduler(optim, opts)
    if opts.lr_scheduler == "steplr":

        scheduler = StepLR(optim, step_size= 10000, gamma=0.5)

    elif opts.lr_scheduler == 'CosineAnnealingWarmRestarts':

        scheduler = CosineAnnealingWarmRestarts(optim, T_0=2000, T_mult=1, eta_min=1e-6,last_epoch=-1)

    elif opts.lr_scheduler == 'LambdaLR':

        scheduler = LambdaLR(optim, lr_lambda=lambda epoch: 0.95 ** epoch)

    else:
        print("Could not input argment LR_scheduler.")
        raise

    return optim, scheduler
