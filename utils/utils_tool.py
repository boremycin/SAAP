import cv2
import numpy as np
import matplotlib.pyplot as plt # type: ignore
import math
import torch


def plt_show_pic(img):
    plt.figure()
    plt.imshow(img.astype(np.uint8))
    plt.show()

def plt_show_img(name, img_src):
    plt.figure()
    plt.title(name)
    plt.imshow(img_src)
    plt.show()


def plt_show_GrayImage(img):
    plt.figure()
    plt.imshow(img.astype(np.uint8), cmap="gray")
    plt.show()


def rgb2gray(rgb):
    return np.dot(rgb[:, :, :3], [0.2125, 0.7154, 0.0721])


def gray_pad_zeros(img, kernel_size):
    w, h = img.shape
    pad_num = (kernel_size - 1) // 2
    new_img = np.zeros((w + 2 * pad_num, h + 2 * pad_num))
    new_img[pad_num:-pad_num, pad_num:-pad_num] = img[:, :]
    # print(w, h, kernel_size,pad_num)
    return new_img


def find_mid_value(img, x, y, kernelsize):
    value_array = np.zeros((kernelsize, kernelsize))
    broaden_value = (kernelsize - 1) // 2
    value_array[:, :] = img[x - broaden_value:x + broaden_value + 1, y - broaden_value:y + broaden_value + 1]
    return np.median(value_array)


def mid_value_filter(img, kernel_size):
    w, h = img.shape
    img_padded = gray_pad_zeros(img, kernel_size)
    broaden_value = (kernel_size - 1) // 2
    img_filtered = np.zeros((w, h))
    for i in range(w):
        for j in range(h):
            img_filtered[i][j] = find_mid_value(img_padded, i + broaden_value, j + broaden_value, kernel_size)
    # plt_show_GrayImage(img_filtered)
    return img_filtered


def dft_1d(array_one_dimension):
    w, = array_one_dimension.shape
    mid_arr = np.zeros((w, w), dtype=complex)
    for i in range(w):
        for j in range(w):
            thta = -2 * math.pi * (j / w) * i
            mid_arr[j][i] = complex(math.cos(thta), math.sin(thta))
    dft_array = array_one_dimension @ mid_arr
    return dft_array


def dft_2d(array_td):
    w0, h0 = array_td.shape
    td_padded = np.zeros((2 * w0, 2 * h0), dtype=complex)
    td_padded[:w0, :h0] = array_td
    # print(td_padded)
    w = 2 * w0
    h = 2 * h0
    mid_array = np.zeros((w, h), dtype=complex)
    for u in range(w):
        for v in range(h):
            mid_uv_array = np.zeros((w, h), dtype=complex)
            for x in range(w):
                for y in range(h):
                    theta = -2 * math.pi * ((u * x) / w + (v * y) / h)
                    mid_uv_array[x][y] = complex(td_padded[x][y] * math.cos(theta), td_padded[x][y] * math.sin(theta))
            mid_array[u][v] = mid_uv_array.sum()
    return mid_array


def dft_2d_shift(array_td):
    w0, h0 = array_td.shape
    td_padded = np.zeros((2 * w0, 2 * h0), dtype=complex)
    td_padded[:w0, :h0] = array_td
    # print(td_padded)
    w = 2 * w0
    h = 2 * h0
    mid_array = np.zeros((w, h), dtype=complex)
    for u in range(w):
        for v in range(h):
            mid_uv_array = np.zeros((w, h), dtype=complex)
            for x in range(w):
                for y in range(h):
                    theta = -2 * math.pi * ((u * x) / w + (v * y) / h)
                    mid_uv_array[x][y] = complex(td_padded[x][y] * math.cos(theta) * pow(-1, x + y),
                                                 td_padded[x][y] * math.sin(theta) * pow(-1, x + y))
            mid_uv_array_gpu = torch.from_numpy(mid_uv_array)
            result = torch.sum(mid_uv_array_gpu, dtype=torch.cfloat)
            mid_array[u][v] = result.numpy()
            #mid_array[u][v] = mid_uv_array.sum()
            print(u, v)
    return mid_array


def bgr2gray(cv_img):
    rgb = cv_img[:, :, ::-1]
    return np.dot(rgb[:, :, :3], [0.2125, 0.7154, 0.0721])


def plt_show_cv_img(img, name=" "):
    show_img = img[:, :, ::-1]
    plt.figure()
    plt.title(name)
    plt.imshow(show_img.astype(np.uint8))
    plt.show()

def img_to_one(img):
    pix_max = np.max(img)
    img_one = img / pix_max
    return img_one

def bright_judge(pic_path):
    flag = 0
    pic_gray = bgr2gray(cv2.imread(pic_path))
    hist = np.bincount((pic_gray.astype(np.uint8)).ravel(), minlength=256)
    mean = 0
    for i in range(len(hist)):
        mean += i * hist[i] / np.sum(hist)
    #print(mean)
    if mean >= 50:
        flag = 1
    else:
        flag = 0
    return flag


def line_row_statics(img_src):
    h, w = img_src.shape
    img_src = np.clip(img_src,175,255)
    img_src[img_src == 175] = 0
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