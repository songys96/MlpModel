import os
import time
import csv
# import copy     # ch9
# import wave     # ch11
# import cv2      # ch 12

import numpy as np
import pandas as pd

from matplotlib import pyplot as plt
from PIL import Image

# from IPython.core.display import HTML   # ch 14

# ----------------------------------------------------------
# 수학 공식 관련 함수
def relu(x):
    return np.maximum(x,0)

def relu_derv(y):
    return np.sign(y)

def sigmoid(x):
    return np.exp(-relu(-x)) / (1.0 + np.exp(-np.abs(x)))

def sigmoid_derv(y):
    return y(1-y)

def sigmoid_cross_entropy_with_logits(z, x):
    return relu(x) - x * z + np.log(1 + np.exp(-np.abs(x)))

def sigmoid_cross_entropy_with_logits_derv(z, x):
    return -z + sigmoid(x)


def tanh(x):
    return 2 * sigmoid(2*x) - 1

def tanh_derv(y):
    return (1.0 + y) * (1.0 - y)

def softmax(x):
    max_elem = np.max(x, axis = 1)
    diff = (x.transpose() - max_elem).transpose()
    exp = np.exp(diff)
    sum_exp = np.sum(exp, axis = 1)
    probs = (exp.transpose() / sum_exp).transpose()
    return probs

def softmax_cross_entropy_with_logits(labels, logits):
    probs = softmax(logits)
    return -np.sum(labels * np.log(probs+1.0e-10), axis=1)

def softmax_cross_entropy_with_logits_derv(labels, logits):
    return softmax(logits) - labels



# ----------------------------------------------------------

def load_csv(path, skip_header= True):
    with open(path) as csvfile:
        csvreader = csv.reader(csvfile)
        headers = None
        if skip_header: headers = next(csvreader, None)
        rows = []
        for row in csvreader:
            rows.append(row)

    return rows, headers

def onehot(xs, cnt):
    return np.eye(cnt)[np.array(xs).astype(int)]

def vector_to_str(x, fmt='%.2f', max_cnt=0):
    """
    벡터처리 함수
    """
    if max_cnt == 0 or len(x) <= max_cnt:
        return '[' + ','.join([fmt]*len(x)) % tuple(x) + ']'
    v = x[0:max_cnt]
    return '[' + ','.join([fmt]*len(v)) % tuple(v) + ',...]'

def load_image_pixels(imagepath, resolution, input_shape):
    """
    이미지 입력 함수
    """
    img = Image.open(imagepath)
    resized = img.resize(resolution)
    return np.array(resized).reshape(input_shape)

def draw_images_horz(xs, image_shape=None):
    """
    이미지 입출력 함수
    """
    show_cnt = len(xs)
    fig, axes = plt.subplots(1, show_cnt, figsize=(5,5))
    for n in range(show_cnt):
        img = xs[n]
        if image_shape:
            x3d = img.reshape(image_shape)
            img = Image.fromarray(np.uint8(x3d))
        axes[n].imshow(img)
        axes[n].axis('off')
    plt.draw()
    plt.show()

def show_select_results(est, ans, target_names, max_cnt=0):
    """
    선택분류 결과 출력 함수
    """
    for n in range(len(est)):
        pstr = vector_to_str(100*est[n], '%.2f', max_cnt)
        estr = target_names[np.argmax(est[n])]
        astr = target_names[np.argmax(ans[n])]
        rstr = '0'
        if estr != astr: rstr = 'X'
        print(f'추정확률 분포 {pstr} => 추정 {estr} | 정답 {astr} => {rstr}')

def list_dir(path):
    filenames = os.listdir(path)
    if '.DS_Store' in filenames:
        filenames.remove('.DS_Store')
    filenames.sort()
    return filenames