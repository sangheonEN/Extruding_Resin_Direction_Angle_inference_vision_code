"""
semantic segmentataion loss function
https://github.com/qubvel/segmentation_models.pytorch/tree/master/segmentation_models_pytorch/losses 
https://github.com/doiken23/focal_segmentation
https://github.com/shruti-jadon/Semantic-Segmentation-Loss-Functions/blob/29c03bb44127c963c9ba3ad0345c789307a5653e/loss_functions.py#L70
1. Cross entropy 2d

2. Dice loss

3. AsymmetricLossMultiLabel

"""

import torch.nn.functional as F
import torch.nn as nn
import torch
from typing import Optional, List
from functools import partial
from torch.nn.modules.loss import _Loss
import math
import numpy as np
from sklearn.preprocessing import MinMaxScaler


#: Loss binary mode suppose you are solving binary segmentation task.
#: That mean yor have only one class which pixels are labled as **1**,
#: the rest pixels are background and labeled as **0**.
#: Target mask shape - (N, H, W), model output mask shape (N, 1, H, W).
BINARY_MODE: str = "binary"

#: Loss multiclass mode suppose you are solving multi-**class** segmentation task.
#: That mean you have *C = 1..N* classes which have unique label values,
#: classes are mutually exclusive and all pixels are labeled with theese values.
#: Target mask shape - (N, H, W), model output mask shape (N, C, H, W).
MULTICLASS_MODE: str = "multiclass"

#: Loss multilabel mode suppose you are solving multi-**label** segmentation task.
#: That mean you have *C = 1..N* classes which pixels are labeled as **1**,
#: classes are not mutually exclusive and each class have its own *channel*,
#: pixels in each channel which are not belong to class labeled as **0**.
#: Target mask shape - (N, C, H, W), model output mask shape (N, C, H, W).
MULTILABEL_MODE: str = "multilabel"


def cross_entropy2d(input, target, weight=None):
    """Softmax + Negative Log Likelihood
       input: (n, c, h, w), target: (n, h, w)
       log_p: (n, c, h, w)
       log_p: (n*h*w, c)
       target: (n*h*w,)
    """
    n, c, h, w = input.size()
    log_p = F.log_softmax(input, dim=1)
    log_p = log_p.transpose(1, 2).transpose(2, 3).contiguous()
    log_p = log_p[target.view(n, h, w, 1).repeat(1, 1, 1, c) >= 0]
    log_p = log_p.view(-1, c)
    mask = target >= 0
    target = target[mask]
    loss = F.nll_loss(log_p, target, weight=weight, reduction='sum')

    return loss


def ce_my_loss(input, target, weight=None):
    """Softmax + Negative Log Likelihood
       input: (n, c, h, w), target: (n, h, w)
       log_p: (n, c, h, w)
       log_p: (n*h*w, c)
       target: (n*h*w,)
    """
    n, c, h, w = input.size()
    eps = 1e-8

    # input_soft : (B, C, H, W)
    softmax_logits = F.softmax(input, dim=1) + eps

    # softmax logits std B, H, W
    sigma = torch.std(softmax_logits, dim=1)

    # loss function weight beta B, H, W
    # beta = -torch.log(sigma) + 1
    # beta = -torch.log(sigma)
    beta = (1/torch.exp(sigma)) + 1
    beta = beta.unsqueeze(dim=1).repeat(1, input.shape[1], 1, 1)

    log_p = F.log_softmax(input, dim=1)
    log_p = log_p.transpose(1, 2).transpose(2, 3).contiguous()
    log_p = log_p[target.view(n, h, w, 1).repeat(1, 1, 1, c) >= 0]
    log_p = log_p.view(-1, c)
    mask = target >= 0
    target = target[mask]
    loss = F.nll_loss(log_p, target, weight=weight, reduction='sum')


    # if reduction == 'none':
    #     # loss : (B, H, W)
    #     loss = loss_tmp
    # elif reduction == 'mean':
    #     # loss : scalar
    #     loss = torch.mean(torch.mean(loss_tmp, dim=0))
    # elif reduction == 'sum':
    #     # loss : scalar
    #     loss = torch.sum(torch.mean(loss_tmp, dim=0))
    # else:
    #     raise NotImplementedError(f"Invalid reduction mode: {reduction}")


    return loss


def cross_entropy_effective_number2d(input, target):
    """Softmax + Negative Log Likelihood
       input: (n, c, h, w), target: (n, h, w)
       log_p: (n, c, h, w)
       log_p: (n*h*w, c)
       target: (n*h*w,)
    """
    eps = 1e-8

    # create the labels one hot tensor
    # target_one_hot : (B, C, H, W)
    target_one_hot = label_to_one_hot_label(target.long(), num_classes=input.shape[1], device=input.device,
                                            dtype=input.dtype, ignore_index=30)

    # weights balanced effective_num
    "flags: effective_num or min_max"
    balance_weights = balance_weights_f(target_one_hot, flags = 'effective_num')

    cross_entropy = nn.CrossEntropyLoss(weight=balance_weights, reduction= 'sum').cuda()
    loss = cross_entropy(input, target.squeeze())

    return loss


class Focal_Effective_Square_Sqrt_Loss(nn.Module):
    r"""Criterion that computes Focal loss.

    According to :cite:`lin2018focal`, the Focal loss is computed as follows:

    .. math:

        FL(p_t) = -alpha_t(1 - p_t)^{gamma}, log(p_t)

    Where:
       - :math:`p_t` is the model's estimated probability for each class.

    Args:
        alpha: Weighting factor :math:`\alpha \in [0, 1]`.
        gamma: Focusing parameter :math:`\gamma >= 0`.
        reduction: Specifies the reduction to apply to the
          output: ``'none'`` | ``'mean'`` | ``'sum'``. ``'none'``: no reduction
          will be applied, ``'mean'``: the sum of the output will be divided by
          the number of elements in the output, ``'sum'``: the output will be
          summed.
        eps: Scalar to enforce numerical stabiliy.

    Shape:
        - Input: :math:`(N, C, *)` where C = number of classes.
        - Target: :math:`(N, *)` where each value is
          :math:`0 ≤ targets[i] ≤ C−1`.

    Example:
        # >>> N = 5  # num_classes
        # >>> kwargs = {"alpha": 0.5, "gamma": 2.0, "reduction": 'mean'}
        # >>> criterion = FocalLoss(**kwargs)
        # >>> input = torch.randn(1, N, 3, 5, requires_grad=True)
        # >>> target = torch.empty(1, 3, 5, dtype=torch.long).random_(N)
        # >>> output = criterion(input, target)
        # >>> output.backward()
    """

    def __init__(self, alpha, gamma=2.0, reduction='mean', eps=1e-8, ignore_index=30, balancing=True, flag_ss = 'square', epoch = 0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.eps = eps
        self.ignore_index = ignore_index
        self.balancing = balancing
        self.flag_ss = flag_ss
        self.epoch = epoch

    def forward(self, input, target):
        return focal_effective_square_sqrt_loss(input, target, self.alpha, self.gamma, self.reduction, self.eps,
                                                self.ignore_index, self.balancing, self.flag_ss, self.epoch)


# https://github.com/zhezh/focalloss/blob/master/focalloss.py
def focal_effective_square_sqrt_loss(input, target, alpha, gamma, reduction, eps, ignore_index, balancing, flag_ss, epoch):
    r"""Criterion that computes Focal loss.

    According to :cite:`lin2018focal`, the Focal loss is computed as follows:

    .. math::

        \text{FL}(p_t) = -\alpha_t (1 - p_t)^{\gamma} \, \text{log}(p_t)

    Where:
       - :math:`p_t` is the model's estimated probability for each class.

    Args:
        input: logits tensor with shape :math:`(N, C, *)` where C = number of classes.
        target: labels tensor with shape :math:`(N, *)` where each value is :math:`0 ≤ targets[i] ≤ C−1`.
        alpha: Weighting factor :math:`\alpha \in [0, 1]`.
        gamma: Focusing parameter :math:`\gamma >= 0`.
        reduction: Specifies the reduction to apply to the
          output: ``'none'`` | ``'mean'`` | ``'sum'``. ``'none'``: no reduction
          will be applied, ``'mean'``: the sum of the output will be divided by
          the number of elements in the output, ``'sum'``: the output will be
          summed.
        eps: Scalar to enforce numerical stabiliy.

    Return:
        the computed loss.

    Example:
        # >>> N = 5  # num_classes
        # >>> input = torch.randn(1, N, 3, 5, requires_grad=True)
        # >>> target = torch.empty(1, 3, 5, dtype=torch.long).random_(N)
        # >>> output = focal_loss(input, target, alpha=0.5, gamma=2.0, reduction='mean')
        # >>> output.backward()
    """
    if not isinstance(input, torch.Tensor):
        raise TypeError(f"Input type is not a torch.Tensor. Got {type(input)}")

    if not len(input.shape) >= 2:
        raise ValueError(f"Invalid input shape, we expect BxCx*. Got: {input.shape}")

    if input.size(0) != target.size(0):
        raise ValueError(f'Expected input batch_size ({input.size(0)}) to match target batch_size ({target.size(0)}).')

    # input : (B, C, H, W)
    n = input.size(0)  # B

    # out_sie : (B, H, W)
    out_size = (n,) + input.size()[2:]

    # input : (B, C, H, W)
    # target : (B, H, W)
    if target.shape[0] == 1:
        target = target.squeeze()
        target = target.unsqueeze(dim=0)
    else:
        target = target.squeeze()
    if target.size()[1:] != input.size()[2:]:
        raise ValueError(f'Expected target size {out_size}, got {target.size()}')

    if not input.device == target.device:
        raise ValueError(f"input and target must be in the same device. Got: {input.device} and {target.device}")

    if isinstance(alpha, float):
        pass
    elif isinstance(alpha, np.ndarray):
        alpha = torch.from_numpy(alpha)
        # alpha : (B, C, H, W)
        alpha = alpha.view(-1, len(alpha), 1, 1).expand_as(input)
    elif isinstance(alpha, torch.Tensor):
        # alpha : (B, C, H, W)
        alpha = alpha.view(-1, len(alpha), 1, 1).expand_as(input)

        # compute softmax over the classes axis
    # input_soft : (B, C, H, W)
    input_soft = F.softmax(input, dim=1) + eps

    # create the labels one hot tensor
    # target_one_hot : (B, C, H, W)
    target_one_hot = label_to_one_hot_label(target.long(), num_classes=input.shape[1], device=input.device,
                                            dtype=input.dtype, ignore_index=ignore_index)

    # compute the actual focal loss
    weight = torch.pow(1.0 - input_soft, gamma)

    # alpha, weight, input_soft : (B, C, H, W)
    # focal : (B, C, H, W)
    focal = -alpha * weight * torch.log(input_soft)

    # weights balanced effective_num
    "flags: effective_num or min_max"
    balance_weights = balance_weights_f(target_one_hot, flags = 'effective_num')

    # square or sqrt weights
    if flag_ss == 'square':
        square_sqrt_balance_weights = torch.square(balance_weights)
    else:
        square_sqrt_balance_weights = torch.sqrt(balance_weights)

    print(f"\n square_effective_num_weights: {square_sqrt_balance_weights}\n")

    # if class num is 0, balance weights became inf
    # if (balancing == False) or (epoch < 10):
    if balancing == False:
        # loss_tmp : (B, H, W)
        torch.nan_to_num(focal)
        loss_tmp = torch.sum(target_one_hot * focal, dim=1)

        loss_flat = loss_tmp.flatten()
        target_flat = target.flatten()

    else:

        torch.nan_to_num(focal)

        # effective weight balancing matrix multiply 행렬 연산으로 weights를 곱해줘야지! 왜 faltten()해서 하냐!
        target_one_hot = target_one_hot.cpu() * balance_weights.reshape(input.shape[1], 1, 1)

        loss_tmp = target_one_hot * focal

        # 과거에 flatten으로 계산함 target class and focal loss pixel wise matching and balance_weights multiply
        # loss_flat = loss_tmp.flatten()
        # target_flat = target.flatten()
        #
        # for i in range(len(balance_weights)):
        #     loss_flat[target_flat==i] = loss_flat[target_flat==i] * balance_weights[i]
        #
        # # print(f"\n mean loss_flat[target_flat==0]: {torch.mean(loss_flat[target_flat==0])}\n")
        # # print(f"\n mean loss_flat[target_flat==1]: {torch.mean(loss_flat[target_flat==1])}\n")
        # # print(f"\n mean loss_flat[target_flat==2]: {torch.mean(loss_flat[target_flat==2])}\n")
        # # print(f"\n mean loss_flat[target_flat==3]: {torch.mean(loss_flat[target_flat==3])}\n")
        #
        # loss_tmp = loss_flat

        # loss_tmp : (B, H, W) batch dim으로 합치기
    if reduction == 'none':
        # loss : (B, H, W)
        loss = loss_tmp
    elif reduction == 'mean':
        # loss : scalar
        loss = torch.mean(torch.mean(loss_tmp, dim=0))
    elif reduction == 'sum':
        # loss : scalar
        loss = torch.sum(torch.mean(loss_tmp, dim=0))
    else:
        raise NotImplementedError(f"Invalid reduction mode: {reduction}")

    return loss


class Focal_Effective_Loss(nn.Module):
    r"""Criterion that computes Focal loss.

    According to :cite:`lin2018focal`, the Focal loss is computed as follows:

    .. math:

        FL(p_t) = -alpha_t(1 - p_t)^{gamma}, log(p_t)

    Where:
       - :math:`p_t` is the model's estimated probability for each class.

    Args:
        alpha: Weighting factor :math:`\alpha \in [0, 1]`.
        gamma: Focusing parameter :math:`\gamma >= 0`.
        reduction: Specifies the reduction to apply to the
          output: ``'none'`` | ``'mean'`` | ``'sum'``. ``'none'``: no reduction
          will be applied, ``'mean'``: the sum of the output will be divided by
          the number of elements in the output, ``'sum'``: the output will be
          summed.
        eps: Scalar to enforce numerical stabiliy.

    Shape:
        - Input: :math:`(N, C, *)` where C = number of classes.
        - Target: :math:`(N, *)` where each value is
          :math:`0 ≤ targets[i] ≤ C−1`.

    Example:
        # >>> N = 5  # num_classes
        # >>> kwargs = {"alpha": 0.5, "gamma": 2.0, "reduction": 'mean'}
        # >>> criterion = FocalLoss(**kwargs)
        # >>> input = torch.randn(1, N, 3, 5, requires_grad=True)
        # >>> target = torch.empty(1, 3, 5, dtype=torch.long).random_(N)
        # >>> output = criterion(input, target)
        # >>> output.backward()
    """

    def __init__(self, alpha, gamma=2.0, reduction='mean', eps=1e-8, ignore_index=30, balancing=True):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.eps = eps
        self.ignore_index = ignore_index
        self.balancing = balancing

    def forward(self, input, target):
        return focal_effective_loss(input, target, self.alpha, self.gamma, self.reduction, self.eps, self.ignore_index, self.balancing)


# https://github.com/zhezh/focalloss/blob/master/focalloss.py
def focal_effective_loss(input, target, alpha, gamma, reduction, eps, ignore_index, balancing):
    r"""Criterion that computes Focal loss.

    According to :cite:`lin2018focal`, the Focal loss is computed as follows:

    .. math::

        \text{FL}(p_t) = -\alpha_t (1 - p_t)^{\gamma} \, \text{log}(p_t)

    Where:
       - :math:`p_t` is the model's estimated probability for each class.

    Args:
        input: logits tensor with shape :math:`(N, C, *)` where C = number of classes.
        target: labels tensor with shape :math:`(N, *)` where each value is :math:`0 ≤ targets[i] ≤ C−1`.
        alpha: Weighting factor :math:`\alpha \in [0, 1]`.
        gamma: Focusing parameter :math:`\gamma >= 0`.
        reduction: Specifies the reduction to apply to the
          output: ``'none'`` | ``'mean'`` | ``'sum'``. ``'none'``: no reduction
          will be applied, ``'mean'``: the sum of the output will be divided by
          the number of elements in the output, ``'sum'``: the output will be
          summed.
        eps: Scalar to enforce numerical stabiliy.

    Return:
        the computed loss.

    Example:
        # >>> N = 5  # num_classes
        # >>> input = torch.randn(1, N, 3, 5, requires_grad=True)
        # >>> target = torch.empty(1, 3, 5, dtype=torch.long).random_(N)
        # >>> output = focal_loss(input, target, alpha=0.5, gamma=2.0, reduction='mean')
        # >>> output.backward()
    """
    if not isinstance(input, torch.Tensor):
        raise TypeError(f"Input type is not a torch.Tensor. Got {type(input)}")

    if not len(input.shape) >= 2:
        raise ValueError(f"Invalid input shape, we expect BxCx*. Got: {input.shape}")

    if input.size(0) != target.size(0):
        raise ValueError(f'Expected input batch_size ({input.size(0)}) to match target batch_size ({target.size(0)}).')

    # input : (B, C, H, W)
    n = input.size(0)  # B

    # out_sie : (B, H, W)
    out_size = (n,) + input.size()[2:]

    # input : (B, C, H, W)
    # target : (B, H, W)
    if target.shape[0] == 1:
        target = target.squeeze()
        target = target.unsqueeze(dim=0)
    else:
        target = target.squeeze()
    if target.size()[1:] != input.size()[2:]:
        raise ValueError(f'Expected target size {out_size}, got {target.size()}')

    if not input.device == target.device:
        raise ValueError(f"input and target must be in the same device. Got: {input.device} and {target.device}")

    if isinstance(alpha, float):
        pass
    elif isinstance(alpha, np.ndarray):
        alpha = torch.from_numpy(alpha)
        # alpha : (B, C, H, W)
        alpha = alpha.view(-1, len(alpha), 1, 1).expand_as(input)
    elif isinstance(alpha, torch.Tensor):
        # alpha : (B, C, H, W)
        alpha = alpha.view(-1, len(alpha), 1, 1).expand_as(input)

        # compute softmax over the classes axis
    # input_soft : (B, C, H, W)
    input_soft = F.softmax(input, dim=1) + eps

    # create the labels one hot tensor
    # target_one_hot : (B, C, H, W)
    target_one_hot = label_to_one_hot_label(target.long(), num_classes=input.shape[1], device=input.device,
                                            dtype=input.dtype, ignore_index=ignore_index)

    # compute the actual focal loss
    weight = torch.pow(1.0 - input_soft, gamma)

    # alpha, weight, input_soft : (B, C, H, W)
    # focal : (B, C, H, W)
    focal = -alpha * weight * torch.log(input_soft)

    # weights balanced effective_num
    # shape class num "flags: effective_num or min_max"
    balance_weights = balance_weights_f(target_one_hot, flags = 'effective_num')
    print(f"\n effective_num_weights: {balance_weights}\n")

    # if class num is 0, balance weights became inf
    if balancing == False:
        # loss_tmp : (B, H, W)
        torch.nan_to_num(focal)
        loss_tmp = torch.sum(target_one_hot * focal, dim=1)
    else:

        torch.nan_to_num(focal)

        # effective weight balancing matrix multiply 행렬 연산으로 weights를 곱해줘야지! 왜 faltten()해서 하냐!
        target_one_hot = target_one_hot.cpu() * balance_weights.reshape(input.shape[1], 1, 1)

        loss_tmp = target_one_hot * focal.cpu()

        # 과거에 flatten으로 계산함 target class and focal loss pixel wise matching and balance_weights multiply
        # loss_flat = loss_tmp.flatten()
        # target_flat = target.flatten()
        #
        # for i in range(len(balance_weights)):
        #     loss_flat[target_flat==i] = loss_flat[target_flat==i] * balance_weights[i]
        #
        # # print(f"\n mean loss_flat[target_flat==0]: {torch.mean(loss_flat[target_flat==0])}\n")
        # # print(f"\n mean loss_flat[target_flat==1]: {torch.mean(loss_flat[target_flat==1])}\n")
        # # print(f"\n mean loss_flat[target_flat==2]: {torch.mean(loss_flat[target_flat==2])}\n")
        # # print(f"\n mean loss_flat[target_flat==3]: {torch.mean(loss_flat[target_flat==3])}\n")
        #
        # loss_tmp = loss_flat

    # loss_tmp : (B, H, W) batch dim으로 합치기
    if reduction == 'none':
        # loss : (B, H, W)
        loss = loss_tmp
    elif reduction == 'mean':
        # loss : scalar
        loss = torch.mean(torch.mean(loss_tmp, dim=0))
    elif reduction == 'sum':
        # loss : scalar
        loss = torch.sum(torch.mean(loss_tmp, dim=0))
    else:
        raise NotImplementedError(f"Invalid reduction mode: {reduction}")

    return loss



class FocalLoss(nn.Module):
    r"""Criterion that computes Focal loss.

    According to :cite:`lin2018focal`, the Focal loss is computed as follows:

    .. math:

        FL(p_t) = -alpha_t(1 - p_t)^{gamma}, log(p_t)

    Where:
       - :math:`p_t` is the model's estimated probability for each class.

    Args:
        alpha: Weighting factor :math:`\alpha \in [0, 1]`.
        gamma: Focusing parameter :math:`\gamma >= 0`.
        reduction: Specifies the reduction to apply to the
          output: ``'none'`` | ``'mean'`` | ``'sum'``. ``'none'``: no reduction
          will be applied, ``'mean'``: the sum of the output will be divided by
          the number of elements in the output, ``'sum'``: the output will be
          summed.
        eps: Scalar to enforce numerical stabiliy.

    Shape:
        - Input: :math:`(N, C, *)` where C = number of classes.
        - Target: :math:`(N, *)` where each value is
          :math:`0 ≤ targets[i] ≤ C−1`.

    Example:
        # >>> N = 5  # num_classes
        # >>> kwargs = {"alpha": 0.5, "gamma": 2.0, "reduction": 'mean'}
        # >>> criterion = FocalLoss(**kwargs)
        # >>> input = torch.randn(1, N, 3, 5, requires_grad=True)
        # >>> target = torch.empty(1, 3, 5, dtype=torch.long).random_(N)
        # >>> output = criterion(input, target)
        # >>> output.backward()
    """

    def __init__(self, alpha, gamma=2.0, reduction='mean', eps=1e-8, ignore_index=30, balancing=True):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.eps = eps
        self.ignore_index = ignore_index
        self.balancing = balancing

    def forward(self, input, target):
        return focal_loss(input, target, self.alpha, self.gamma, self.reduction, self.eps, self.ignore_index)


# https://github.com/zhezh/focalloss/blob/master/focalloss.py
def focal_loss(input, target, alpha, gamma, reduction, eps, ignore_index):
    r"""Criterion that computes Focal loss.

    According to :cite:`lin2018focal`, the Focal loss is computed as follows:

    .. math::

        \text{FL}(p_t) = -\alpha_t (1 - p_t)^{\gamma} \, \text{log}(p_t)

    Where:
       - :math:`p_t` is the model's estimated probability for each class.

    Args:
        input: logits tensor with shape :math:`(N, C, *)` where C = number of classes.
        target: labels tensor with shape :math:`(N, *)` where each value is :math:`0 ≤ targets[i] ≤ C−1`.
        alpha: Weighting factor :math:`\alpha \in [0, 1]`.
        gamma: Focusing parameter :math:`\gamma >= 0`.
        reduction: Specifies the reduction to apply to the
          output: ``'none'`` | ``'mean'`` | ``'sum'``. ``'none'``: no reduction
          will be applied, ``'mean'``: the sum of the output will be divided by
          the number of elements in the output, ``'sum'``: the output will be
          summed.
        eps: Scalar to enforce numerical stabiliy.

    Return:
        the computed loss.

    Example:
        # >>> N = 5  # num_classes
        # >>> input = torch.randn(1, N, 3, 5, requires_grad=True)
        # >>> target = torch.empty(1, 3, 5, dtype=torch.long).random_(N)
        # >>> output = focal_loss(input, target, alpha=0.5, gamma=2.0, reduction='mean')
        # >>> output.backward()
    """
    if not isinstance(input, torch.Tensor):
        raise TypeError(f"Input type is not a torch.Tensor. Got {type(input)}")

    if not len(input.shape) >= 2:
        raise ValueError(f"Invalid input shape, we expect BxCx*. Got: {input.shape}")

    if input.size(0) != target.size(0):
        raise ValueError(f'Expected input batch_size ({input.size(0)}) to match target batch_size ({target.size(0)}).')

    # input : (B, C, H, W)
    n = input.size(0)  # B

    # out_sie : (B, H, W)
    out_size = (n,) + input.size()[2:]

    # input : (B, C, H, W)
    # target : (B, H, W)
    if target.shape[0] == 1:
        target = target.squeeze()
        target = target.unsqueeze(dim=0)
    else:
        target = target.squeeze()
    if target.size()[1:] != input.size()[2:]:
        raise ValueError(f'Expected target size {out_size}, got {target.size()}')

    if not input.device == target.device:
        raise ValueError(f"input and target must be in the same device. Got: {input.device} and {target.device}")

    if isinstance(alpha, float):
        pass
    elif isinstance(alpha, np.ndarray):
        alpha = torch.from_numpy(alpha)
        # alpha : (B, C, H, W)
        alpha = alpha.view(-1, len(alpha), 1, 1).expand_as(input)
    elif isinstance(alpha, torch.Tensor):
        # alpha : (B, C, H, W)
        alpha = alpha.view(-1, len(alpha), 1, 1).expand_as(input)

        # compute softmax over the classes axis
    # input_soft : (B, C, H, W)
    input_soft = F.softmax(input, dim=1) + eps

    sigma = torch.std(input_soft, dim=1)

    # create the labels one hot tensor
    # target_one_hot : (B, C, H, W)
    target_one_hot = label_to_one_hot_label(target.long(), num_classes=input.shape[1], device=input.device,
                                            dtype=input.dtype, ignore_index=ignore_index)

    # compute the actual focal loss
    weight = torch.pow(1.0 - input_soft, gamma)

    # alpha, weight, input_soft : (B, C, H, W)
    # focal : (B, C, H, W)
    focal = -alpha * weight * torch.log(input_soft)

    torch.nan_to_num(focal)

    # loss_tmp = torch.sum(target_one_hot * focal, dim=1)

    loss_tmp = target_one_hot * focal

    if reduction == 'none':
        # loss : (B, H, W)
        loss = loss_tmp
    elif reduction == 'mean':
        # loss : scalar
        loss = torch.mean(torch.mean(loss_tmp, dim=0))
    elif reduction == 'sum':
        # loss : scalar
        loss = torch.sum(torch.mean(loss_tmp, dim=0))
    else:
        raise NotImplementedError(f"Invalid reduction mode: {reduction}")

    return loss, sigma


class Hard_Easy_Loss(nn.Module):
    r"""Criterion that computes Focal loss.

    According to :cite:`lin2018focal`, the Focal loss is computed as follows:

    .. math:

        FL(p_t) = -alpha_t(1 - p_t)^{gamma}, log(p_t)

    Where:
       - :math:`p_t` is the model's estimated probability for each class.

    Args:
        alpha: Weighting factor :math:`\alpha \in [0, 1]`.
        gamma: Focusing parameter :math:`\gamma >= 0`.
        reduction: Specifies the reduction to apply to the
          output: ``'none'`` | ``'mean'`` | ``'sum'``. ``'none'``: no reduction
          will be applied, ``'mean'``: the sum of the output will be divided by
          the number of elements in the output, ``'sum'``: the output will be
          summed.
        eps: Scalar to enforce numerical stabiliy.

    Shape:
        - Input: :math:`(N, C, *)` where C = number of classes.
        - Target: :math:`(N, *)` where each value is
          :math:`0 ≤ targets[i] ≤ C−1`.

    Example:
        # >>> N = 5  # num_classes
        # >>> kwargs = {"alpha": 0.5, "gamma": 2.0, "reduction": 'mean'}
        # >>> criterion = FocalLoss(**kwargs)
        # >>> input = torch.randn(1, N, 3, 5, requires_grad=True)
        # >>> target = torch.empty(1, 3, 5, dtype=torch.long).random_(N)
        # >>> output = criterion(input, target)
        # >>> output.backward()
    """

    def __init__(self, alpha, gamma=2.0, reduction='mean', beta=1.5, threshold=0.2, eps=1e-8, ignore_index=30, balancing=True):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.eps = eps
        self.ignore_index = ignore_index
        self.balancing = balancing
        self.beta = beta
        self.threshold = threshold

    def forward(self, input, target):
        return hard_easy_f(input, target, self.alpha, self.gamma, self.reduction, self.eps, self.ignore_index, self.beta, self.threshold)


# https://github.com/zhezh/focalloss/blob/master/focalloss.py
def hard_easy_f(input, target, alpha, gamma, reduction, eps, ignore_index, beta, threshold):
    r"""Criterion that computes Focal loss.

    According to :cite:`lin2018focal`, the Focal loss is computed as follows:

    .. math::

        \text{FL}(p_t) = -\alpha_t (1 - p_t)^{\gamma} \, \text{log}(p_t)

    Where:
       - :math:`p_t` is the model's estimated probability for each class.

    Args:
        input: logits tensor with shape :math:`(N, C, *)` where C = number of classes.
        target: labels tensor with shape :math:`(N, *)` where each value is :math:`0 ≤ targets[i] ≤ C−1`.
        alpha: Weighting factor :math:`\alpha \in [0, 1]`.
        gamma: Focusing parameter :math:`\gamma >= 0`.
        reduction: Specifies the reduction to apply to the
          output: ``'none'`` | ``'mean'`` | ``'sum'``. ``'none'``: no reduction
          will be applied, ``'mean'``: the sum of the output will be divided by
          the number of elements in the output, ``'sum'``: the output will be
          summed.
        eps: Scalar to enforce numerical stabiliy.

    Return:
        the computed loss.

    Example:
        # >>> N = 5  # num_classes
        # >>> input = torch.randn(1, N, 3, 5, requires_grad=True)
        # >>> target = torch.empty(1, 3, 5, dtype=torch.long).random_(N)
        # >>> output = focal_loss(input, target, alpha=0.5, gamma=2.0, reduction='mean')
        # >>> output.backward()
    """
    if not isinstance(input, torch.Tensor):
        raise TypeError(f"Input type is not a torch.Tensor. Got {type(input)}")

    if not len(input.shape) >= 2:
        raise ValueError(f"Invalid input shape, we expect BxCx*. Got: {input.shape}")

    if input.size(0) != target.size(0):
        raise ValueError(f'Expected input batch_size ({input.size(0)}) to match target batch_size ({target.size(0)}).')

    # input : (B, C, H, W)
    n = input.size(0)  # B

    # out_sie : (B, H, W)
    out_size = (n,) + input.size()[2:]

    # input : (B, C, H, W)
    # target : (B, H, W)
    if target.shape[0] == 1:
        target = target.squeeze()
        target = target.unsqueeze(dim=0)
    else:
        target = target.squeeze()
    if target.size()[1:] != input.size()[2:]:
        raise ValueError(f'Expected target size {out_size}, got {target.size()}')

    if not input.device == target.device:
        raise ValueError(f"input and target must be in the same device. Got: {input.device} and {target.device}")

    if isinstance(alpha, float):
        pass
    elif isinstance(alpha, np.ndarray):
        alpha = torch.from_numpy(alpha)
        # alpha : (B, C, H, W)
        alpha = alpha.view(-1, len(alpha), 1, 1).expand_as(input)
    elif isinstance(alpha, torch.Tensor):
        # alpha : (B, C, H, W)
        alpha = alpha.view(-1, len(alpha), 1, 1).expand_as(input)

        # compute softmax over the classes axis
    """
    softmax logits max, min threshold, beta 
    # # input_soft : (B, C, H, W)
    # softmax_logits = F.softmax(input, dim=1) + eps
    # 
    # logits_max, max_indices = torch.max(softmax_logits, dim=1)
    # logits_min, min_indices = torch.min(softmax_logits, dim=1)
    # 
    # # theta = B, H, W
    # theta = logits_max - logits_min
    # 
    # # hard and easy sample index extract
    # hard = np.where(theta.cpu() <= threshold)
    # easy = np.where(theta.cpu() > threshold)
    # 
    # # create the labels one hot tensor
    # # target_one_hot : (B, C, H, W)
    # target_one_hot = label_to_one_hot_label(target.long(), num_classes=input.shape[1], device=input.device,
    #                                         dtype=input.dtype, ignore_index=ignore_index)
    # 
    # # compute the actual focal loss
    # weight = torch.pow(1.0 - softmax_logits, gamma)
    # 
    # # alpha, weight, input_soft : (B, C, H, W)
    # # focal : (B, C, H, W)
    # focal = -alpha * weight * torch.log(softmax_logits)
    # 
    # # hard sample focal loss * beta weights
    # focal[hard[0], :, hard[1], hard[2]] *= beta
    """

    """
    softmax * onehot 확률이 threshold 이상인건 easy, threshold 이하인건 hard로 하여 hard sample에 대해 beta >= 1.0 라는 값으로 높여줌 
    """

    # input_soft : (B, C, H, W)
    softmax_logits = F.softmax(input, dim=1) + eps

    # create the labels one hot tensor
    # target_one_hot : (B, C, H, W)
    target_one_hot = label_to_one_hot_label(target.long(), num_classes=input.shape[1], device=input.device,
                                            dtype=input.dtype, ignore_index=ignore_index)

    # B, H, W
    softmax_logits_probability = torch.sum(softmax_logits * target_one_hot, dim=1)

    # hard[0]: B, hard[1]: H, hard[2]: W
    hard = np.where(softmax_logits_probability.cpu() <= threshold)
    easy = np.where(softmax_logits_probability.cpu() > threshold)

    # compute the actual focal loss
    weight = torch.pow(1.0 - softmax_logits, gamma)

    # alpha, weight, input_soft : (B, C, H, W)
    # focal : (B, C, H, W)
    focal = -alpha * weight * torch.log(softmax_logits)
    focal[hard[0], :, hard[1], hard[2]] *= beta

    torch.nan_to_num(focal)

    # loss_tmp = torch.sum(target_one_hot * focal, dim=1)
    loss_tmp = target_one_hot * focal

    if reduction == 'none':
        # loss : (B, H, W)
        loss = loss_tmp
    elif reduction == 'mean':
        # loss : scalar
        loss = torch.mean(torch.mean(loss_tmp, dim=0))
    elif reduction == 'sum':
        # loss : scalar
        loss = torch.sum(torch.mean(loss_tmp, dim=0))
    else:
        raise NotImplementedError(f"Invalid reduction mode: {reduction}")

    return loss

class LASD(nn.Module):
    r"""Criterion that computes Focal loss.

    According to :cite:`lin2018focal`, the Focal loss is computed as follows:

    .. math:

        FL(p_t) = -alpha_t(1 - p_t)^{gamma}, log(p_t)

    Where:
       - :math:`p_t` is the model's estimated probability for each class.

    Args:
        alpha: Weighting factor :math:`\alpha \in [0, 1]`.
        gamma: Focusing parameter :math:`\gamma >= 0`.
        reduction: Specifies the reduction to apply to the
          output: ``'none'`` | ``'mean'`` | ``'sum'``. ``'none'``: no reduction
          will be applied, ``'mean'``: the sum of the output will be divided by
          the number of elements in the output, ``'sum'``: the output will be
          summed.
        eps: Scalar to enforce numerical stabiliy.

    Shape:
        - Input: :math:`(N, C, *)` where C = number of classes.
        - Target: :math:`(N, *)` where each value is
          :math:`0 ≤ targets[i] ≤ C−1`.

    Example:
        # >>> N = 5  # num_classes
        # >>> kwargs = {"alpha": 0.5, "gamma": 2.0, "reduction": 'mean'}
        # >>> criterion = FocalLoss(**kwargs)
        # >>> input = torch.randn(1, N, 3, 5, requires_grad=True)
        # >>> target = torch.empty(1, 3, 5, dtype=torch.long).random_(N)
        # >>> output = criterion(input, target)
        # >>> output.backward()
    """

    def __init__(self, alpha, gamma=2.0, reduction='mean', eps=1e-8, ignore_index=30, balancing=True):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.eps = eps
        self.ignore_index = ignore_index
        self.balancing = balancing


    def forward(self, input, target):
        return lasd_f(input, target, self.alpha, self.gamma, self.reduction, self.eps, self.ignore_index)


# https://github.com/zhezh/focalloss/blob/master/focalloss.py
def lasd_f(input, target, alpha, gamma, reduction, eps, ignore_index):
    r"""Criterion that computes Focal loss.

    According to :cite:`lin2018focal`, the Focal loss is computed as follows:

    .. math::

        \text{FL}(p_t) = -\alpha_t (1 - p_t)^{\gamma} \, \text{log}(p_t)

    Where:
       - :math:`p_t` is the model's estimated probability for each class.

    Args:
        input: logits tensor with shape :math:`(N, C, *)` where C = number of classes.
        target: labels tensor with shape :math:`(N, *)` where each value is :math:`0 ≤ targets[i] ≤ C−1`.
        alpha: Weighting factor :math:`\alpha \in [0, 1]`.
        gamma: Focusing parameter :math:`\gamma >= 0`.
        reduction: Specifies the reduction to apply to the
          output: ``'none'`` | ``'mean'`` | ``'sum'``. ``'none'``: no reduction
          will be applied, ``'mean'``: the sum of the output will be divided by
          the number of elements in the output, ``'sum'``: the output will be
          summed.
        eps: Scalar to enforce numerical stabiliy.

    Return:
        the computed loss.

    Example:
        # >>> N = 5  # num_classes
        # >>> input = torch.randn(1, N, 3, 5, requires_grad=True)
        # >>> target = torch.empty(1, 3, 5, dtype=torch.long).random_(N)
        # >>> output = focal_loss(input, target, alpha=0.5, gamma=2.0, reduction='mean')
        # >>> output.backward()
    """
    if not isinstance(input, torch.Tensor):
        raise TypeError(f"Input type is not a torch.Tensor. Got {type(input)}")

    if not len(input.shape) >= 2:
        raise ValueError(f"Invalid input shape, we expect BxCx*. Got: {input.shape}")

    if input.size(0) != target.size(0):
        raise ValueError(f'Expected input batch_size ({input.size(0)}) to match target batch_size ({target.size(0)}).')

    # input : (B, C, H, W)
    n = input.size(0)  # B

    # out_sie : (B, H, W)
    out_size = (n,) + input.size()[2:]

    # input : (B, C, H, W)
    # target : (B, H, W)
    if target.shape[0] == 1:
        target = target.squeeze()
        target = target.unsqueeze(dim=0)
    else:
        target = target.squeeze()
    if target.size()[1:] != input.size()[2:]:
        raise ValueError(f'Expected target size {out_size}, got {target.size()}')

    if not input.device == target.device:
        raise ValueError(f"input and target must be in the same device. Got: {input.device} and {target.device}")

    if isinstance(alpha, float):
        pass
    elif isinstance(alpha, np.ndarray):
        alpha = torch.from_numpy(alpha)
        # alpha : (B, C, H, W)
        alpha = alpha.view(-1, len(alpha), 1, 1).expand_as(input)
    elif isinstance(alpha, torch.Tensor):
        # alpha : (B, C, H, W)
        alpha = alpha.view(-1, len(alpha), 1, 1).expand_as(input)

        # compute softmax over the classes axis
    # input_soft : (B, C, H, W)
    softmax_logits = F.softmax(input, dim=1) + eps

    # softmax logits std B, H, W
    sigma = torch.std(softmax_logits, dim=1)

    # min max scaling
    sigma = min_max_scale_f(sigma)

    # loss function weight beta B, H, W
    # beta = -torch.log(sigma) + 1
    # beta = -torch.log(sigma)
    beta = (1/torch.exp(sigma)) + 1

    # create the labels one hot tensor
    # target_one_hot : (B, C, H, W)
    target_one_hot = label_to_one_hot_label(target.long(), num_classes=input.shape[1], device=input.device,
                                            dtype=input.dtype, ignore_index=ignore_index)

    # compute the actual focal loss
    weight = torch.pow(1.0 - softmax_logits, gamma)

    # alpha, weight, input_soft : (B, C, H, W)
    # focal : (B, C, H, W)
    focal = -alpha * weight * torch.log(softmax_logits)

    # hard sample focal loss * beta weights
    beta = beta.unsqueeze(dim=1).repeat(1, input.shape[1], 1, 1)
    focal = focal * beta.to('cuda')

    torch.nan_to_num(focal)

    # loss_tmp = torch.sum(target_one_hot * focal, dim=1) B, H, W
    loss_tmp = target_one_hot * focal

    if reduction == 'none':
        # loss : (B, H, W)
        loss = loss_tmp
    elif reduction == 'mean':
        # loss : scalar
        loss = torch.mean(torch.mean(loss_tmp, dim=0))

        # list_ = list()
        #
        # for loss_batch in loss_tmp:
        #     batch_mean = torch.mean(loss_batch)
        #     list_.append(batch_mean)
        #
        # loss = sum(list_) / len(list_)

    elif reduction == 'sum':
        # loss : scalar
        loss = torch.sum(torch.mean(loss_tmp, dim=0))

        # list_ = list()
        #
        # for loss_batch in loss_tmp:
        #     batch_mean = torch.mean(loss_batch)
        #     list_.append(batch_mean)
        #
        # loss = sum(list_)

    else:
        raise NotImplementedError(f"Invalid reduction mode: {reduction}")

    return loss, sigma

def min_max_scale_f(sig):

    b, h, w = sig.shape[0], sig.shape[1], sig.shape[2]

    logits = torch.flatten(sig, start_dim=0, end_dim=-1)

    # MinMaxScaler 선언 및 Fitting
    mMscaler = MinMaxScaler()
    mMscaler.fit(logits.cpu().detach().numpy().reshape(-1, 1))

    # 데이터 변환 16, 1
    mMscaled_data = mMscaler.transform(logits.cpu().detach().numpy().reshape(-1, 1))

    mMscaled_data = mMscaled_data.reshape(b, h, w)
    mMscaled_data = torch.Tensor(mMscaled_data)

    return mMscaled_data



def balance_weights_f(target_one_hot, flags):
    """
    flags: effective_num or min_max
    eps: eps is prevent inf, nan because if class num is zero, loss value is inf
    """
    eps = float(1e-3)

    if flags == 'effective_num':
        class0_num = torch.sum(target_one_hot[:, 0, :, :] >= 1.0)
        class1_num = torch.sum(target_one_hot[:, 1, :, :] >= 1.0)
        class2_num = torch.sum(target_one_hot[:, 2, :, :] >= 1.0)
        class3_num = torch.sum(target_one_hot[:, 3, :, :] >= 1.0)

        beta = float((class0_num + class1_num + class2_num + class3_num - 1) / (
                    class0_num + class1_num + class2_num + class3_num))

        samples_per_cls = [class0_num, class1_num, class2_num, class3_num]
        effective_num = (1.0 - (np.power(beta, samples_per_cls)) + eps) / (1.0 - beta)
        weights = 1.0 / effective_num
        weights = weights.astype(np.float32)
        weights = torch.from_numpy(weights)
        weights = weights * (len(samples_per_cls) / (torch.sum(weights)))

    else:
        cls_num_list = list()
        max_min_list = list()

        class0_num = torch.sum(target_one_hot[:, 0, :, :] >= 1.0)
        class1_num = torch.sum(target_one_hot[:, 1, :, :] >= 1.0)
        class2_num = torch.sum(target_one_hot[:, 2, :, :] >= 1.0)
        class3_num = torch.sum(target_one_hot[:, 3, :, :] >= 1.0)

        cls_num_list.append(class0_num)
        cls_num_list.append(class1_num)
        cls_num_list.append(class2_num)
        cls_num_list.append(class3_num)

        cls_num_array = np.array(cls_num_list, dtype=np.int)
        max = cls_num_array.max()
        min = cls_num_array.min()

        max_min_list.append(((cls_num_array[0] - min) / (max - min)) + eps)
        max_min_list.append(((cls_num_array[1] - min) / (max - min)) + eps)
        max_min_list.append(((cls_num_array[2] - min) / (max - min)) + eps)
        max_min_list.append(((cls_num_array[3] - min) / (max - min)) + eps)

        max_min_array = np.array(max_min_list, dtype=np.int)

        weights = 1 - max_min_array

    return weights



def label_to_one_hot_label(
        labels: torch.Tensor,
        num_classes: int,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
        eps: float = 1e-6,
        ignore_index=255,
) -> torch.Tensor:
    r"""Convert an integer label x-D tensor to a one-hot (x+1)-D tensor.

    Args:
        labels: tensor with labels of shape :math:`(N, *)`, where N is batch size.
          Each value is an integer representing correct classification.
        num_classes: number of classes in labels.
        device: the desired device of returned tensor.
        dtype: the desired data type of returned tensor.

    Returns:
        the labels in one hot tensor of shape :math:`(N, C, *)`,

    Examples:
        # >>> labels = torch.LongTensor([
                [[0, 1],
                [2, 0]]
            ])
        # >>> one_hot(labels, num_classes=3)
        tensor([[[[1.0000e+00, 1.0000e-06],
                  [1.0000e-06, 1.0000e+00]],

                 [[1.0000e-06, 1.0000e+00],
                  [1.0000e-06, 1.0000e-06]],

                 [[1.0000e-06, 1.0000e-06],
                  [1.0000e+00, 1.0000e-06]]]])

    """
    shape = labels.shape
    # one hot : (B, C=ignore_index+1, H, W)
    one_hot = torch.zeros((shape[0], ignore_index + 1) + shape[1:], device=device, dtype=dtype)

    # labels : (B, H, W)
    # labels.unsqueeze(1) : (B, C=1, H, W)
    # one_hot : (B, C=ignore_index+1, H, W)
    one_hot = one_hot.scatter_(1, labels.unsqueeze(1), 1.0) + eps

    # ret : (B, C=num_classes, H, W)
    ret = torch.split(one_hot, [num_classes, ignore_index + 1 - num_classes], dim=1)[0]

    return ret