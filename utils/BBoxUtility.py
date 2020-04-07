import numpy as np
import torch
from torchvision import ops


class BBoxUtility(object):
    def __init__(self, priors=None, overlap_threshold=0.7, ignore_threshold=0.3,
                 nms_thresh=0.7, top_k=300):
        self.priors = priors
        self.num_priors = 0 if priors is None else len(priors)
        self.overlap_threshold = overlap_threshold
        self.ignore_threshold = ignore_threshold
        self._nms_thresh = nms_thresh
        self._top_k = top_k

    @property
    def nms_thresh(self):
        return self._nms_thresh

    @property
    def top_k(self):
        return self._top_k

    # box为标注框
    def iou(self, box):
        # 计算出每个真实框与所有的先验框的iou
        # 判断真实框与先验框的重合情况
        inter_upleft = np.maximum(self.priors[:, :2], box[:2])
        inter_botright = np.minimum(self.priors[:, 2:4], box[2:])

        inter_wh = inter_botright - inter_upleft
        inter_wh = np.maximum(inter_wh, 0)
        inter = inter_wh[:, 0] * inter_wh[:, 1]
        # 真实框的面积
        area_true = (box[2] - box[0]) * (box[3] - box[1])
        # 先验框的面积
        area_gt = (self.priors[:, 2] - self.priors[:, 0]) * (self.priors[:, 3] - self.priors[:, 1])
        # 计算iou
        union = area_true + area_gt - inter

        iou = inter / union
        return iou

    def encode_box(self, box, return_iou=True):
        iou = self.iou(box)
        # 4+1，前4个是框的偏移量编码，最后一个是iou
        encoded_box = np.zeros((self.num_priors, 4 + return_iou))

        # 找到每一个真实框，重合程度较高的先验框
        assign_mask = iou > self.overlap_threshold
        if not assign_mask.any():
            assign_mask[iou.argmax()] = True
        # iou阈值大于0.7，列表的最后一个元素填进iou
        if return_iou:
            encoded_box[:, -1][assign_mask] = iou[assign_mask]

        # 找到对应的先验框
        assigned_priors = self.priors[assign_mask]
        # 逆向编码，将真实框转化为FasterRCNN预测结果的格式
        # 先计算真实框的中心与长宽
        box_center = 0.5 * (box[:2] + box[2:])
        box_wh = box[2:] - box[:2]
        # 再计算重合度较高的先验框的中心与长宽
        assigned_priors_center = 0.5 * (assigned_priors[:, :2] +
                                        assigned_priors[:, 2:4])
        assigned_priors_wh = (assigned_priors[:, 2:4] -
                              assigned_priors[:, :2])

        # 逆向求取FasterRCNN应该有的预测结果
        encoded_box[:, :2][assign_mask] = box_center - assigned_priors_center
        encoded_box[:, :2][assign_mask] /= assigned_priors_wh
        encoded_box[:, :2][assign_mask] *= 4

        encoded_box[:, 2:4][assign_mask] = np.log(box_wh / assigned_priors_wh)
        encoded_box[:, 2:4][assign_mask] *= 4
        return encoded_box.ravel()

    def ignore_box(self, box):
        iou = self.iou(box)
        ignored_box = np.zeros((self.num_priors, 1))
        # 找到每一个真实框，重合程度较高的先验框
        assign_mask = (iou > self.ignore_threshold) & (iou < self.overlap_threshold)
        if not assign_mask.any():
            assign_mask[iou.argmax()] = True
        ignored_box[:, 0][assign_mask] = iou[assign_mask]
        return ignored_box.ravel()

    # box ,ground truth标记框的左上、右下的小数形式
    # anchors 生成锚框的小数形式，
    def assign_boxes(self, boxes, anchors):
        self.num_priors = len(anchors)
        self.priors = anchors
        assignment = np.zeros((self.num_priors, 4 + 1))
        assignment[:, 4] = 0.0
        if len(boxes) == 0:
            return assignment

        # 计算需要忽略的先验框
        # 对每一个真实框都进行iou计算,判断是否要忽略，ingored_boxes是一个len(boxes) * 锚框个数  的二维向量
        # 如果锚框x和box y的iou在0.3-0.7之间，ingored_boxes[y,x] = 对应的iou（>0）
        ingored_boxes = np.apply_along_axis(self.ignore_box, 1, boxes[:, :4])
        # 取重合程度最大的先验框，并且获取这个先验框的index
        ingored_boxes = ingored_boxes.reshape(-1, self.num_priors, 1)
        # (num_priors)
        # 只要锚框与任一真实框之间满足忽略要求，对应位置的值就不是0，max就会取出
        ignore_iou = ingored_boxes[:, :, 0].max(axis=0)
        # (num_priors)
        ignore_iou_mask = ignore_iou > 0
        assignment[:, 4][ignore_iou_mask] = -1  # -1表示忽略

        # (n, num_priors, 5)
        encoded_boxes = np.apply_along_axis(self.encode_box, 1, boxes[:, :4])
        # 每一个真实框的编码后的值，和iou
        # (n, num_priors)
        encoded_boxes = encoded_boxes.reshape(-1, self.num_priors, 5)

        # 取重合程度最大的先验框，并且获取这个先验框的index
        # (num_priors)
        # best_iou 存放最好的ioU
        best_iou = encoded_boxes[:, :, -1].max(axis=0)
        # (num_priors)
        # 存放最好的iou对应的下标
        best_iou_idx = encoded_boxes[:, :, -1].argmax(axis=0)
        # (num_priors)
        best_iou_mask = best_iou > 0
        # 某个先验框它属于哪个真实框
        best_iou_idx = best_iou_idx[best_iou_mask]

        assign_num = len(best_iou_idx)
        # 保留重合程度最大的先验框的应该有的预测结果
        # 哪些先验框存在真实框
        encoded_boxes = encoded_boxes[:, best_iou_mask, :]

        assignment[:, :4][best_iou_mask] = encoded_boxes[best_iou_idx, np.arange(assign_num), :4]
        # 4代表为背景的概率，为0
        assignment[:, 4][best_iou_mask] = 1
        # 通过assign_boxes我们就获得了，输入进来的这张图片，应该有的预测结果是什么样子的
        return assignment

    def decode_boxes(self, mbox_loc, mbox_priorbox):
        print("mbox_loc.shape = {}".format(mbox_loc))
        print("mbox_priorbox.shape = {}".format(mbox_priorbox))
        # 获得先验框的宽与高
        prior_width = mbox_priorbox[:, 2] - mbox_priorbox[:, 0]
        prior_height = mbox_priorbox[:, 3] - mbox_priorbox[:, 1]

        # 获得先验框的中心点
        prior_center_x = 0.5 * (mbox_priorbox[:, 2] + mbox_priorbox[:, 0])
        prior_center_y = 0.5 * (mbox_priorbox[:, 3] + mbox_priorbox[:, 1])

        # 真实框距离先验框中心的xy轴偏移情况
        decode_bbox_center_x = mbox_loc[:, 0] * prior_width / 4
        decode_bbox_center_x += prior_center_x
        decode_bbox_center_y = mbox_loc[:, 1] * prior_height / 4
        decode_bbox_center_y += prior_center_y

        # 真实框的宽与高的求取
        decode_bbox_width = np.exp(mbox_loc[:, 2] / 4)
        decode_bbox_width *= prior_width
        decode_bbox_height = np.exp(mbox_loc[:, 3] / 4)
        decode_bbox_height *= prior_height

        # 获取真实框的左上角与右下角
        decode_bbox_xmin = decode_bbox_center_x - 0.5 * decode_bbox_width
        decode_bbox_ymin = decode_bbox_center_y - 0.5 * decode_bbox_height
        decode_bbox_xmax = decode_bbox_center_x + 0.5 * decode_bbox_width
        decode_bbox_ymax = decode_bbox_center_y + 0.5 * decode_bbox_height

        # 真实框的左上角与右下角进行堆叠
        decode_bbox = np.concatenate((decode_bbox_xmin[:, None],
                                      decode_bbox_ymin[:, None],
                                      decode_bbox_xmax[:, None],
                                      decode_bbox_ymax[:, None]), axis=-1)
        # 防止超出0与1
        decode_bbox = np.minimum(np.maximum(decode_bbox, 0.0), 1.0)
        print("decode_bbox = {}".format(decode_bbox))
        return decode_bbox

    # predictions rpn网络的输出，元组（元素为Tensor对象）
    # mbox_priorbox：anchors numpy

    def detection_out(self, predictions, mbox_priorbox, keep_top_k=300,
                      confidence_threshold=0.5):

        # 网络预测的结果
        # 置信度
        mbox_conf = torch.from_numpy(predictions[0])
        mbox_loc = torch.from_numpy(predictions[1])
        mbox_priorbox = torch.from_numpy(mbox_priorbox)
        print("mbox_conf.shape = {}".format(mbox_conf.shape))
        print("mbox_loc.shape = {}".format(mbox_loc.shape))
        # 先验框
        # mbox_priorbox = torch.from_numpy(mbox_priorbox)
        results = []
        # 对每一个图片进行处理
        for i in range(mbox_loc.shape[0]):
            print("index of getexample = {}".format(i))
            results.append([])  # ？？？？
            decode_bbox = self.decode_boxes(mbox_loc[i], mbox_priorbox)
            c_confs = mbox_conf[i, :, 0]
            c_confs_m = c_confs > confidence_threshold
            if c_confs[c_confs_m].shape[0] > 0:
                # 取出得分高于confidence_threshold的框
                boxes_to_process = torch.from_numpy(decode_bbox[c_confs_m]).float()  # 置信度高的框，左上，右下，小数形式
                confs_to_process = c_confs[c_confs_m].float()
                print("boxes_to_process.shape={}".format(boxes_to_process.shape))
                print("confs_to_process.shape={}".format(confs_to_process.shape))
                idx = ops.nms(boxes=boxes_to_process, scores=confs_to_process, iou_threshold=0.8)
                # print("confs_to_process[index]= {}".format(confs_to_process[idx]))
                good_boxes = boxes_to_process[idx].numpy()

                # 添加进result里
                results[-1].extend(good_boxes)
                # 选出置信度最大的keep_top_k个
                results[-1] = results[-1][:keep_top_k]

        # 获得，在所有预测结果里面，置信度比较高的框
        # 还有，利用先验框和RPN的预测结果，处理获得了真实框（预测框）的位置
        return np.array(results)
