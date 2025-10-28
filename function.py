import numpy as np
import time
import os
import csv
import matplotlib.pyplot as plt
import io
from numpy.core.defchararray import title

import random

from query_strategies.strategy import Strategy
import torch.nn.functional as F
from query_strategies import *
# *时间相关函数
# -获取时分秒
def get_hms_time(sec):
    s_hour = 3600
    s_min = 60
    h = int(sec/s_hour)
    m = int((sec%s_hour)/s_min)
    s = int(sec%s_min)
    return h, m, s
    

# *根据标记，统计样本比例
# idxs:一个数组，里面的元素对应着这一次选取的样本下标；labels：所有的标签
# 输入的labels是tensor向量，返回一个numpy数组
def get_mnist_prop(idxs, labels, len, count_class):
    prop_lbs = np.zeros(count_class)
    count_lbs = np.zeros(count_class)
    for i in range(0,len):
        count_lbs[labels[idxs[i]]] += 1
    for i in range(0,count_class):
        prop_lbs[i] = count_lbs[i]/len
    return prop_lbs, count_lbs


# *路径相关函数
# 修改 out-path，这是用来存储最终的log、csv以及对应的图片的
def get_results_dir(args):
    # args.
    time_start = time.localtime()
    path_results_tmp = '{}-{}-{}'.format(args.model_name, args.dataset, time.strftime(".%Y-%m-%d-%H-%M-%S", time_start))
    path_results_fin = os.path.join(args.out_path, path_results_tmp)
    if not os.path.exists(path_results_fin):
        os.makedirs(path_results_fin)
    args.out_path = path_results_fin
    return

def get_init_seg(args, idxs_tmp, n_init_pool, X,Y_Ptr, SEED):
    np.random.seed(SEED)
    method_init = args.method_init
    idxs_use = []

    if method_init == 'RS':
        idxs_use = idxs_tmp[:n_init_pool]
    elif method_init == 'area_seg':

        area_list_init = []
        for i in range(Y_Ptr.size(0)):  # 对每个样本进行处理
            # 对每个样本，找到伪标签大于0.5的区域
            mask = Y_Ptr[i] > 0.5
            area = mask.sum().item()  # 计算该区域的面积（大于0.5的像素数量）
            area_list_init.append(area)

        sorted_indices = np.argsort(area_list_init)[-n_init_pool:]

        idxs_use = sorted_indices

    return idxs_use





