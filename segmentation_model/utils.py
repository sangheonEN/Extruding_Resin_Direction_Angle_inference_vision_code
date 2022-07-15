import torch
import numpy as np
import os
import os.path as osp
import cv2
import yaml
import albumentations as A
import matplotlib.pyplot as plt


def data_sort_list(input_path, mask_path):

    def sort_function(f): return int(''.join(filter(str.isdigit, f)))

    input_list = os.listdir(input_path)
    input_list = [file for file in input_list if file.endswith("png")]
    input_list.sort(key=sort_function)

    mask_list = os.listdir(mask_path)
    mask_list = [file for file in mask_list if file.endswith('png')]
    mask_list.sort(key=sort_function)

    return input_list, mask_list


def get_log_dir(model_name, cfg):

    name = 'MODEL-%s' % (model_name)

    log_dir = osp.join('logs', name)
    if not osp.exists(log_dir):
        os.makedirs(log_dir)
    with open(osp.join(log_dir, 'config.yaml'), 'w') as f:
        yaml.safe_dump(cfg, f, default_flow_style=False)
    return log_dir


def get_config():
    return {
        # same configuration as original work
        # https://github.com/shelhamer/fcn.berkeleyvision.org
        1:
        dict(
            max_iteration=12500,
            lr=1.0e-7,
            momentum=0.99,
            weight_decay=0.0005
            # interval_validate=4000
        )
    }


def get_cuda(cuda, _id):
    if not cuda:
        return torch.device('cpu')
    else:
        return torch.device('cuda:{}'.format(_id))


def run_fromfile(model, img_file, cuda):
    img_torch = img_file
    img_torch = img_torch.to(cuda)
    model.eval()
    with torch.no_grad():
        score = model(img_torch)
        return score


def _fast_hist(label_true, label_pred, n_class):
    mask = (label_true >= 0) & (label_true < n_class)
    hist = np.bincount(n_class * label_true[mask].astype(int) +
                       label_pred[mask],
                       minlength=n_class**2).reshape(n_class, n_class)
    return hist


def label_accuracy_score(label_trues, label_preds, n_class):
    """Returns accuracy score evaluation result.

      - overall accuracy
      - mean accuracy
      - mean IU
      - fwavacc

    """
    hist = np.zeros((n_class, n_class))
    for lt, lp in zip(label_trues, label_preds):
        hist += _fast_hist(lt.flatten(), lp.flatten(), n_class)
    acc = np.diag(hist).sum() / hist.sum()
    acc_cls = np.diag(hist) / hist.sum(axis=1)
    acc_cls = np.nanmean(acc_cls)
    iu = np.diag(hist) / (hist.sum(axis=1) + hist.sum(axis=0) - np.diag(hist))
    mean_iu = np.nanmean(iu)
    freq = hist.sum(axis=1) / hist.sum()
    fwavacc = (freq[freq > 0] * iu[freq > 0]).sum()
    return acc, acc_cls, mean_iu, fwavacc


def augmentation_train(inputs_train_feed, masks_train_feed):
    # torch tensor operation기준인 input shape이 (batch, channel, height, width)로 들어오니
    # 이걸 augmentation 하기위해서는 (batch, height, width, channel)로 변환 필요.
    inputs_train_feed = np.array(inputs_train_feed).transpose(0, 2, 3, 1)
    masks_train_feed = np.array(masks_train_feed).transpose(0, 2, 3, 1)

    mean_resnet = np.array([0.485, 0.456, 0.406])
    std_resnet = np.array([0.229, 0.224, 0.225])

    iter_num = inputs_train_feed.shape[0]

    inputs_list = list()
    masks_list = list()

    transform = A.Compose([
        A.Rotate(limit=10, p=0.5)
        # A.HorizontalFlip(p=1)
    ])
    
    flip_flag = False
    
    if np.random.uniform() > 0.5:
        transform2 = A.Compose([
            # A.Rotate(limit=10, p=0.6),
            A.HorizontalFlip(p=1)
        ])
        flip_flag = True
    

    for i in range(iter_num):
        
        transformed = transform(image=inputs_train_feed[i], mask=masks_train_feed[i])       

        input = transformed["image"]
        mask = transformed["mask"]
        
        # vertical flip -> curve class switching
        if flip_flag == True:
            temp = 4
            switch_class_1 = 2
            swtich_class_2 = 3
            transformed2 = transform2(image=inputs_train_feed[i], mask=masks_train_feed[i])
            input = transformed2["image"]
            mask = transformed2["mask"]
            mask[mask == switch_class_1] = temp
            mask[mask == swtich_class_2] = switch_class_1
            mask[mask == temp] = swtich_class_2
            
        input /= 255.
        input -= mean_resnet
        input /= std_resnet

        inputs_list.append(input)
        masks_list.append(mask)

    # torch tensor operation기준인 input shape(batch, channel, height, width)으로 다시 변환.
    img = np.array(inputs_list, dtype=np.float32).transpose(0, 3, 1, 2)
    lbl = np.array(masks_list, dtype=np.int32).transpose(0, 3, 1, 2)

    img = torch.from_numpy(img.copy()).float()
    lbl = torch.from_numpy(lbl.copy()).long()

    return img, lbl


LEE_COLORMAP = [
    [0, 0, 0],
    [255, 255, 255],
    [255, 0, 0],
    [0, 0, 255]
]


def decode_segmap_save(label_mask, n_classes, flag_pred, cnt, plot=False):
    """Decode segmentation class labels into a color image

    Args:
        label_mask (np.ndarray): an (M,N) array of integer values denoting
          the class label at each spatial location.
        plot (bool, optional): whether to show the resulting color image
          in a figure.

    Returns:
        (np.ndarray, optional): the resulting decoded color image.
    """
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
    if plot:
        plt.imshow(rgb)
        plt.show()
    else:
        if flag_pred == True:
            mkdir_f(os.path.join(os.path.dirname(os.path.abspath(
                __file__)), 'inference_result', 'inference_pred'))
            cv2.imwrite(os.path.join(os.path.dirname(os.path.abspath(
                __file__)), 'inference_result', 'inference_pred', '%04d.png' % cnt), rgb)
        else:
            mkdir_f(os.path.join(os.path.dirname(os.path.abspath(
                __file__)), 'inference_result', 'inference_anno'))
            cv2.imwrite(os.path.join(os.path.dirname(os.path.abspath(
                __file__)), 'inference_result', 'inference_anno', '%04d.png' % cnt), rgb)


def mkdir_f(path):

    if not os.path.exists(path):
        os.makedirs(path)

        return

