import numpy as np
import copy


def union(au, bu, area_intersection):
    area_a = (au[2] - au[0]) * (au[3] - au[1])
    area_b = (bu[2] - bu[0]) * (bu[3] - bu[1])
    area_union = area_a + area_b - area_intersection
    return area_union


def intersection(ai, bi):
    x = max(ai[0], bi[0])
    y = max(ai[1], bi[1])
    w = min(ai[2], bi[2]) - x
    h = min(ai[3], bi[3]) - y
    if w < 0 or h < 0:
        return 0
    return w * h


def iou(a, b):
    if a[0] >= a[2] or a[1] >= a[3] or b[0] >= b[2] or b[1] >= b[3]:
        return 0.0

    area_i = intersection(a, b)
    area_u = union(a, b, area_i)

    return float(area_i) / float(area_u + 1e-6)

# R rpn推荐的较好的框，num * (左上、右下)
# all_boxes：真实框的小数形式
def calc_iou(R, config, all_boxes, width, height, num_classes):
    # print(all_boxes)
    print("all_boxes.shape = {}".format(all_boxes.shape))
    bboxes = all_boxes[:, :4]
    gta = np.zeros((len(bboxes), 4))
    for bbox_num, bbox in enumerate(bboxes):
        gta[bbox_num, 0] = int(round(bbox[0] * width / config.rpn_stride))
        gta[bbox_num, 1] = int(round(bbox[1] * height / config.rpn_stride))
        gta[bbox_num, 2] = int(round(bbox[2] * width / config.rpn_stride))
        gta[bbox_num, 3] = int(round(bbox[3] * height / config.rpn_stride))
    x_roi = []
    y_class_num = []
    y_class_regr_coords = []
    print( "gta = {}".format(gta))
    for ix in range(R.shape[0]):
        x1 = R[ix, 0] * width / config.rpn_stride
        y1 = R[ix, 1] * height / config.rpn_stride
        x2 = R[ix, 2] * width / config.rpn_stride
        y2 = R[ix, 3] * height / config.rpn_stride

        x1 = int(round(x1))
        y1 = int(round(y1))
        x2 = int(round(x2))
        y2 = int(round(y2))
        # print("[x1, y1, x2, y2] = {}".format([x1, y1, x2, y2]))
        best_iou = 0.0
        best_bbox = -1
        # rpn推荐的每一个框和真实框之间比较，计算IoU
        for bbox_num in range(len(bboxes)):
            curr_iou = iou([gta[bbox_num, 0], gta[bbox_num, 1], gta[bbox_num, 2], gta[bbox_num, 3]], [x1, y1, x2, y2])
            if curr_iou > best_iou:
                best_iou = curr_iou
                best_bbox = bbox_num
        # print(best_iou)
        if best_iou < config.classifier_min_overlap:
            continue
        else:
            w = x2 - x1
            h = y2 - y1
            x_roi.append([x1, y1, w, h])

            if config.classifier_min_overlap <= best_iou < config.classifier_max_overlap:
                label = num_classes
            elif config.classifier_max_overlap <= best_iou:

                label = int(all_boxes[best_bbox, -1])
                cxg = (gta[best_bbox, 0] + gta[best_bbox, 2]) / 2.0
                cyg = (gta[best_bbox, 1] + gta[best_bbox, 3]) / 2.0

                cx = x1 + w / 2.0
                cy = y1 + h / 2.0

                tx = (cxg - cx) / float(w)
                ty = (cyg - cy) / float(h)
                tw = np.log((gta[best_bbox, 2] - gta[best_bbox, 0]) / float(w))
                th = np.log((gta[best_bbox, 3] - gta[best_bbox, 1]) / float(h))
            else:
                raise RuntimeError
        y_class_num.append([copy.deepcopy(label)])
        # 4*20   20是左上右下
        coords = [0] * 4 * num_classes
        if label != num_classes:
            label_pos = 4 * label
            sx, sy, sw, sh = config.classifier_regr_std
            coords[label_pos:4 + label_pos] = [sx * tx, sy * ty, sw * tw, sh * th]
            y_class_regr_coords.append(copy.deepcopy(coords))
        else:
            y_class_regr_coords.append(copy.deepcopy(coords))

    if len(x_roi) == 0:
        return None, None, None

    X = np.array(x_roi)
    Y1 = np.array(y_class_num)
    Y2 = np.array(y_class_regr_coords)

    return np.expand_dims(X, axis=0), np.expand_dims(Y1, axis=0), np.expand_dims(Y2, axis=0)