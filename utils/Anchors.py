import numpy as np
from utils.Config import Config
from utils.ImageUtil import draw_rect_mul

config = Config()


# 每个网格上的9个先验框
def generate_anchors(sizes=None, ratios=None):
    if sizes is None:
        sizes = config.anchor_box_scales

    if ratios is None:
        ratios = config.anchor_box_ratios

    num_anchors = len(sizes) * len(ratios)

    anchors = np.zeros((num_anchors, 4))

    anchors[:, 2:] = np.tile(sizes, (2, len(ratios))).T

    for i in range(len(ratios)):
        anchors[3 * i:3 * i + 3, 2] = anchors[3 * i:3 * i + 3, 2] * ratios[i][0]
        anchors[3 * i:3 * i + 3, 3] = anchors[3 * i:3 * i + 3, 3] * ratios[i][1]

    anchors[:, 0::2] -= np.tile(anchors[:, 2] * 0.5, (2, 1)).T
    anchors[:, 1::2] -= np.tile(anchors[:, 3] * 0.5, (2, 1)).T
    return anchors


def shift(shape, anchors, stride=config.rpn_stride):
    # 每一个网格的中心
    shift_x = (np.arange(0, shape[0], dtype=float) + 0.5) * stride
    shift_y = (np.arange(0, shape[1], dtype=float) + 0.5) * stride
    shift_x, shift_y = np.meshgrid(shift_x, shift_y)

    shift_x = np.reshape(shift_x, [-1])
    shift_y = np.reshape(shift_y, [-1])

    shifts = np.stack([
        shift_x,
        shift_y,
        shift_x,
        shift_y
    ], axis=0)

    shifts = np.transpose(shifts)

    number_of_anchors = np.shape(anchors)[0]

    k = np.shape(shifts)[0]

    shifted_anchors = np.reshape(anchors, [1, number_of_anchors, 4]) + np.array(np.reshape(shifts, [k, 1, 4]), float)
    shifted_anchors = np.reshape(shifted_anchors, [k * number_of_anchors, 4])
    return shifted_anchors


# 获得先验框 shape ：公用特征层的大小，当输入图片为600 * 600，shape 为38 * 38
# width，height输入图片的宽高
# 输出先验框的左上角和右下角的归一化形式
def get_anchors(shape, width, height):
    anchors = generate_anchors()
    network_anchors = shift(shape, anchors)
    network_anchors[:, 0] = network_anchors[:, 0] / width
    network_anchors[:, 1] = network_anchors[:, 1] / height
    network_anchors[:, 2] = network_anchors[:, 2] / width
    network_anchors[:, 3] = network_anchors[:, 3] / height
    network_anchors = np.clip(network_anchors, 0, 1)
    return network_anchors


if __name__ == '__main__':
    from PIL import Image
    import torchvision.transforms as transforms


    def to_rgb(image):
        rgb_image = Image.new("RGB", image.size)
        rgb_image.paste(image)
        return rgb_image


    __transforms__ = [
        transforms.Resize((600, 600), Image.BICUBIC),
        transforms.ToTensor(),
        # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ]
    transform = transforms.Compose(__transforms__)
    image = Image.open(r"G:\cv\faster-rcnn-keras/VOCdevkit/VOC2007/JPEGImages/2007_000027.jpg")
    if image.mode != "RGB":
        image = to_rgb(image)
    image = transform(image)
    image = image.numpy()
    achors = get_anchors([38, 38], 600, 600) * 600
    _ = np.array(achors[50:100])
    draw_rect_mul(image, _)
