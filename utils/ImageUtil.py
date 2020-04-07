import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from PIL import Image


# 输入image：C * H * W,numpy数据
def draw_rect_one(image, top=0, left=0, width=10, height=10, is_save=False, save_path="./temp.jpg"):
    # np.transpose( xxx,  (2, 0, 1))   # 将 C x H x W 转化为 H x W x C
    image = np.transpose(image, (1, 2, 0))
    fig, ax = plt.subplots(1)
    rect = patches.Rectangle((top, left), width, height, linewidth=1, edgecolor='r', fill=False)
    ax.imshow(image)
    ax.add_patch(rect)
    plt.axis('off')
    if is_save:
        plt.savefig(save_path)
    plt.show()


# 输入image：C * H * W,numpy数据
# rectangles为框的列表，每个框提供左上、右下
def draw_rect_mul(image, rectangles, is_save=False, save_path="./temp.jpg"):
    # np.transpose( xxx,  (2, 0, 1))   # 将 C x H x W 转化为 H x W x C
    image = np.transpose(image, (1, 2, 0))
    fig, ax = plt.subplots(1)
    ax.imshow(image)
    edgecolor = ['r', 'b', 'g','y']
    for index, rec in enumerate(rectangles):
        print("index = {}".format(index) )
        print(rec)
        top, left, right, bottom = rec[0], rec[1], rec[2], rec[3]
        width = right - left
        height = bottom - top
        rect = patches.Rectangle((top, left), width, height, linewidth=1,
                                 edgecolor=edgecolor[index % len(edgecolor)],
                                 fill=False)
        ax.add_patch(rect)
    plt.axis('off')
    if is_save:
        plt.savefig(save_path)
    plt.show()


# 输入image：C * H * W,numpy数据,显示图片
def showImage(image):
    image = np.transpose(image, (1, 2, 0))
    plt.imshow(image)
    plt.axis('off')
    plt.show()


def __to_rgb__(image):
    rgb_image = Image.new("RGB", image.size)
    rgb_image.paste(image)
    return rgb_image


def loadImage(path):
    image = Image.open(path)
    if image.mode != "RGB":
        image = __to_rgb__(image)
    return image


if __name__ == '__main__':
    pass
