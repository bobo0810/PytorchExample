import cv2
import os
from glob import glob
import random
import matplotlib.pyplot as plt
import argparse
from tqdm import tqdm
import numpy as np
import os
cur_path=os.path.abspath(os.path.dirname(__file__))
parser = argparse.ArgumentParser()
parser.add_argument('--images_path', type=str, default="/code/images")  # 图片根路径  
parser.add_argument('--labels_path', type=str, default="/code/labels")  # 标签根路径  标签名字与图片名称对应   
parser.add_argument('--classes_path', type=str, default=cur_path+"/classes.txt") # 类别名称的路径 每行是类别名称，N个类别共N行
parser.add_argument('--output_path', type=str, default=cur_path+"/output") # 输出路径
parser.add_argument('--conf', type=float, default=0.4) # 仅可视化置信度>conf的bbox
parser.add_argument('--vis_conf', type=bool, default=True) # True:显示置信度
arg = parser.parse_args()

colorlist = []
# 5^3种颜色。
for i in range(25, 256, 50):
    for j in range(25, 256, 50):
        for k in range(25, 256, 50):
            colorlist.append((i, j, k))
random.shuffle(colorlist)


def plot_bbox(img_path, img_dir, out_dir, gt,cls2label=None, line_thickness=None):
    img = cv2.imread(os.path.join(img_dir, img_path))
    height, width, _ = img.shape
    tl = line_thickness or round(0.002 * (width + height) / 2) + 1  # line/font thickness

    with open(gt, 'r') as f:
        annotations = f.readlines()
        # print(annotations)
        for ann in annotations:
            ann = list(map(float, ann.split()))
            ann[0] = int(ann[0])
            # print(ann)
            if len(ann) == 6:
                cls, x, y, w, h, conf = ann
                if conf < arg.conf:
                    # thres = 0.5
                    continue
            elif len(ann) == 5:
                cls, x, y, w, h = ann
            color = colorlist[len(colorlist) - cls - 1]

            c1, c2 = (int((x - w / 2) * width), int((y - h / 2) * height)), (
            int((x + w / 2) * width), int((y + h / 2) * height))
            cv2.rectangle(img, c1, c2, color, thickness=tl, lineType=cv2.LINE_AA)

            # # cls label
            tf = max(tl - 1, 1)  # font thickness
            t_size = cv2.getTextSize(cls2label[cls], 0, fontScale=tl / 2, thickness=tf)[0]
            c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
            # cv2.rectangle(img, c1, c2, color, -1, cv2.LINE_AA)  # filled

            if len(ann) == 5 or not arg.vis_conf:
                cv2.putText(img, cls2label[cls], (c1[0], c1[1] - 2), 0, tl / 2, color,thickness=tf, lineType=cv2.LINE_AA)
            elif len(ann) == 6:
                cv2.putText(img, str(cls2label[cls] + str(round(conf, 2))), (c1[0], c1[1] - 2), 0, tl / 2, color,
                            thickness=tf, lineType=cv2.LINE_AA)
    cv2.imwrite(os.path.join(out_dir, img_path), img)


if __name__ == "__main__":
    img_dir = arg.images_path
    GT_dir = arg.labels_path
    out_dir = arg.output_path  
    cls_dir = arg.classes_path
    cls_dict = {}


    if not os.path.exists(img_dir):
        raise Exception("image dir {} do not exist!".format(img_dir))
    if not os.path.exists(cls_dir):
        raise Exception("class dir {} do not exist!".format(cls_dir))
    else:
        with open(cls_dir, 'r') as f:
            classes = f.readlines()
            for i in range(len(classes)):
                cls_dict[i] = classes[i].strip()
            print("class map:", cls_dict)
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)
    if not os.path.exists(GT_dir):
        print(f"WARNNING: {GT_dir} ,GT NOT Available!")
    for each_img in tqdm(os.listdir(img_dir)):
        gt = None
        # 支持jpg、png、jepg等
        suffix=os.path.basename(each_img).split(".")[-1]
        if os.path.exists(os.path.join(GT_dir, each_img.replace(suffix, 'txt'))):
            gt = os.path.join(GT_dir, each_img.replace(suffix, 'txt'))

        if gt:
            plot_bbox(each_img, img_dir, out_dir, gt, cls2label=cls_dict)
        else:
            print(each_img,"no detect")