import torch
from torch import nn
import torch.nn.functional as F
import numpy as np


def smooth_l1(y_true, y_pred, sigma=1.0):
    # y_true [batch_size, num_anchor, 4+1]
    # y_pred [batch_size, num_anchor, 4]

    sigma_squared = sigma ** 2
    regression = y_pred.view(-1, 4)
    regression_target = y_true[:, :, :-1].float()
    regression_target = regression_target.view(-1, 4).float()
    anchor_state = y_true[:, :, -1]

    # 正样本
    indices_for_object = np.array((anchor_state.cpu() == 1).nonzero()).reshape(-1)
    regression = regression[indices_for_object]
    regression_target = regression_target[indices_for_object]

    loss = F.smooth_l1_loss(regression, regression_target)


    return loss


def cls_loss(y_true, y_pred, ratio=3):
    # y_true [batch_size, num_anchor, num_classes]
    # y_pred [batch_size, num_anchor, num_classes]

    y_true = y_true.view(-1, y_true.shape[2]).float()
    y_pred = y_pred.view(-1, y_pred.shape[2]).float()

    labels = y_true  # 维数可能有问题
    anchor_state = y_true[:, -1]  # -1 是需要忽略的, 0 是背景, 1 是存在目标
    print(anchor_state.shape)
    classification = y_pred

    # 找出存在目标的先验框
    indices_for_object = np.array((anchor_state.cpu() == 1).nonzero()).reshape(-1)
    # print("postive = {}".format(indices_for_object.shape[0]))
    labels_for_object = labels[indices_for_object]
    classification_for_object = classification[indices_for_object]
    cls_loss_for_object = F.binary_cross_entropy(classification_for_object, labels_for_object)

    # 找出实际上为背景的先验框
    indices_for_back = np.array((anchor_state.cpu() == 0).nonzero()).reshape(-1)
    # print("back = {}".format(indices_for_back.shape))
    labels_for_back = labels[indices_for_back]
    classification_for_back = classification[indices_for_back]
    cls_loss_for_back = F.binary_cross_entropy(classification_for_back, labels_for_back)

    num_pos = torch.tensor(indices_for_object.shape[0]).float()
    num_pos = torch.max(num_pos, torch.tensor(1).float())
    cls_loss_for_object = cls_loss_for_object / num_pos

    num_neg = torch.tensor(indices_for_back.shape[0]).float()
    num_neg = torch.max(num_neg, torch.tensor(1).float())
    cls_loss_for_back = cls_loss_for_back / num_neg

    loss = cls_loss_for_object + ratio * cls_loss_for_back

    return loss


cross_entropy_loss = nn.CrossEntropyLoss()


def cls_loss_detector(y_true, y_pred):
    return cross_entropy_loss(y_pred, y_true)


def regress_loss_detector(y_true, y_pred, label):
    print("y_true .shape ={}".format(y_true.shape))
    print("y_pred .shape ={}".format(y_pred.shape))
    y_true = y_true.view(y_true.shape[0], -1, 4)
    y_pred = y_pred.view(y_pred.shape[0], -1, 4)
    position = (label != 20).nonzero().reshape(-1)

    # position = np.array(position.nonzero()).reshape(-1)
    print("position = {}".format(position))
    print("y_true .shape ={}".format(y_true.shape))
    print("y_pred .shape ={}".format(y_pred.shape))
    print("y_pred[position][label[position]] = {}".format(y_pred[position,label[position]]))
    print("y_true[position][label[position]] = {}".format(y_true[position,label[position]]))
    loss = F.smooth_l1_loss(input=y_pred[position, label[position]], target=y_true[position, label[position]])
    print("nor_loss = {}".format(loss))
    return loss
