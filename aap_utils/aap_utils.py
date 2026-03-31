import requests
import numpy as np
from PIL import Image
from io import BytesIO
import torch
import pandas as pd
import numpy as np
from torchvision.transforms import transforms
from tqdm import tqdm

from art.estimators.object_detection.pytorch_yolo import PyTorchYolo
from art.attacks.evasion import AdversarialPatchPyTorch
from .loss1 import ComputeLoss
from tqdm.auto import trange
import os
import cv2
from pathlib import Path

def rgb2gray(rgb):
    return np.dot(rgb[:, :, :3], [0.2125, 0.7154, 0.0721])

class Yolo(torch.nn.Module):
    def __init__(self, model,alpha=0.5,beta=0.5):
        super().__init__()
        self.model = model
        self.model.hyp = {'box': 0.05,
                        'obj': 1.0,
                        'cls': 0.5,
                        'anchor_t': 4.0,
                        'cls_pw': 1.0,
                        'obj_pw': 1.0,
                        'fl_gamma': 0.0
                        }
        self.loss_alpha = alpha
        self.loss_beta = beta
        self.compute_loss = ComputeLoss(self.model.model.model,alpha = self.loss_alpha, beta = self.loss_beta )

    def forward(self, x, targets=None):
        if self.training: #将model设定为train后返回梯度信息而不是预测结果
            outputs = self.model.model.model(x)
            loss, loss_items, loss_obj = self.compute_loss(outputs, targets)
            loss_components_dict = {"loss_total": loss} #通过修改此处获得不同项比例不同的loss损失
            loss_components_dict['loss_box'] = loss_items[0]
            loss_components_dict['loss_obj'] = loss_items[1]
            loss_components_dict['loss_cls'] = loss_items[2]
            return loss_components_dict
        else:
            return self.model(x)
        
        
def get_imlist(path):
    return [os.path.join(path, f) for f in os.listdir(path) if f.endswith('.jpg')] 

def get_txtlist(path):
    return [os.path.join(path, f) for f in os.listdir(path) if f.endswith('.txt')]


def plt_show_img(name, img_src):
    plt.figure()
    plt.title(name)
    plt.imshow(img_src)
    plt.show()

def line_row_statics2(img_src):
    h, w = img_src.shape
    #img_src = np.clip(img_src,175,255)
    #img_src[img_src == 175] = 0
    line_statics = np.zeros(w)
    cor_lin = np.arange(w)
    for i in range(w):
        for j in range(h):
            if img_src[i][j]:
                line_statics[i] += 1
    row_statics = np.zeros(h)
    cor_row = np.arange(h)
    for j in range(h):
        for i in range(w):
            if img_src[i][j]:
                row_statics[j] += 1
    x = np.argmax(line_statics)
    y = np.argmax(row_statics)
    #print("line:{}, row:{}".format(x,y))
    lamta = 640/256
    xx = int(x*lamta) - 10
    yy = int(y*lamta) - 10
    return xx,yy

def cv2_ship_seg(img):
    img = np.clip(img,175,255)
    img[img == 175] = 0
    #plt_show_img('1',img)
    blurd_img = cv2.GaussianBlur(img, (7, 7), 0)
    gray_img = cv2.cvtColor(blurd_img, cv2.COLOR_BGR2GRAY)
    #plt_show_img('2',gray_img)
    ret, binary = cv2.threshold(gray_img, 175, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    output = cv2.connectedComponents(binary, connectivity=8, ltype=cv2.CV_32S)
    num_labels = output[0]
    labels = output[1]
    labels_exact = np.zeros((5, 256, 256))
    h, w = np.shape(labels)
    # find the useful layer, 2ed
    for i in range(5):
        for j in range(h):
            for k in range(w):
                if labels[j][k] == i:
                    labels_exact[i][j][k] = labels[j][k]
    colors = []
    for i in range(num_labels):
        b = np.random.randint(0, 256)
        g = np.random.randint(0, 256)
        r = np.random.randint(0, 256)
        colors.append((b, g, r))
    colors[0] = (0, 0, 0)

    h, w = gray_img.shape
    image = np.zeros((h, w, 3), dtype=np.uint8)
    for row in range(h):
        for col in range(w):
            image[row, col] = colors[labels[row, col]]
    #plt_show_img('final',image)
    #print("total components : ", num_labels - 1)
    return_img = rgb2gray(image)
    return return_img



def data_loader_test(src, is_test=False, test_num=None): # attack part of the val dataset
    """
    Two types of information (especially patch loaction info) is returned, from txt label and prejection profile respectively,
    nanmely txt_list_out and loc_list_out
    """
    src_path = Path(src)

    image_dir = src_path / "images"
    label_dir = src_path / "labels"

    txt_files = sorted(list(label_dir.glob("*.txt")))

    if is_test and test_num:
        txt_files = txt_files[:test_num]

    img_list_out = []
    txt_list_out = []
    loc_list_out = []

    pbar = tqdm(txt_files, desc="Loading Data")
    
    for txt_path in pbar:
        img_path = image_dir / f"{txt_path.stem}.jpg"

        if not img_path.exists():
            print(f"\n[Warning] image not found: {img_path}")
            continue
        img_p_str = img_path.as_posix()
        txt_p_str = txt_path.as_posix()

        with open(txt_path, 'r') as f:
            loc_info_list = [line.split() for line in f.readlines()]

        ship_num = len(loc_info_list)
        ship_loc = []
        #print(img_p,'\n',loc_info_list,'\n',ship_num)

        img_mid_raw = cv2.imread(img_p_str)
        if img_mid_raw is None:
            continue

        for j in range(ship_num):
            loc_y = int(256 * float(loc_info_list[j][1]))
            loc_x = int(256 * float(loc_info_list[j][2]))
            loc_h = int(256 * float(loc_info_list[j][3]))
            loc_w = int(256 * float(loc_info_list[j][4]))

            y_start, y_end = max(0, loc_y - loc_h // 2), min(256, loc_y + loc_h // 2)
            x_start, x_end = max(0, loc_x - loc_w // 2), min(256, loc_x + loc_w // 2)
            ship_mid_raw1 = img_mid_raw.copy()

            ship_mid_raw1[x_start:x_end, y_start:y_end, :] = 0
            ship_mid = img_mid_raw - ship_mid_raw1
        
            segged_ship_mid = cv2_ship_seg(ship_mid)
            x1, y1 = line_row_statics2(segged_ship_mid)
            
            if x1 <= 0 or y1 <= 0:
                x1 = int((640 / 256) * loc_x)
                y1 = int((640 / 256) * loc_y)
            
            ship_loc.append([x1, y1])
        img_list_out.append(img_p_str)
        txt_list_out.append(txt_p_str)
        loc_list_out.append(ship_loc)
    return img_list_out, txt_list_out, loc_list_out