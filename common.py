#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@File    :   common.py    
@Author  :   name
@Modify Time      @Version    @Desciption
------------      --------    -----------
2021/4/21 下午3:50    1.0         None
"""
import numpy as np
from sklearn.cluster import KMeans


def iou(list1, list2):
    xx1 = np.maximum(list1[0], list2[0])
    yy1 = np.maximum(list1[1], list2[1])
    xx2 = np.minimum(list1[2], list2[2])
    yy2 = np.minimum(list1[3], list2[3])

    # w = np.maximum(0.0, xx2 - xx1 + 1)
    # h = np.maximum(0.0, yy2 - yy1 + 1)
    w = np.maximum(0.0, xx2 - xx1 + 1)
    h = np.maximum(0.0, yy2 - yy1 + 1)

    inter = w * h
    area1 = (list1[2] - list1[0] + 1) * (list1[3] - list1[1] + 1)
    area2 = (list2[2] - list2[0] + 1) * (list2[3] - list2[1] + 1)
    iou = inter / min(area1, area2)
    w_iou = w / (np.maximum(list1[2], list2[2])-np.minimum(list1[0], list2[0]))
    h_iou = h / (np.maximum(list1[3], list2[3])-np.minimum(list1[1], list2[1]))
    return iou, w_iou, h_iou


def k_mean(new_data, krow, kcol):
    boxes = []
    for cell_info in new_data:
        box = cell_info[2]
        boxes.append(box)
    # row_index, clo_index = judge(boxes)
    # print(row_index, clo_index)
    all_points_row = []
    # all_points_col = []
    for chaos_cell in boxes:
        x1 = float(chaos_cell[0])
        y1 = float(chaos_cell[1])
        x2 = float(chaos_cell[2])
        y2 = float(chaos_cell[3])
        c1 = (x1 + x2) / 2
        # c2 = (y1 + y2) / 2
        all_points_row.append([int(x1), int(x2), int(c1)])
        # all_points_col.append([int(y1), int(y2), int(c2)])
    kmeans_col = KMeans(n_clusters=kcol).fit(all_points_row)
    # print(kmeans_col)
    # kmeans_row = KMeans(n_clusters=krow).fit(all_points_col)
    # print(kmeans_row)
    # l_row = kmeans_row.n_clusters
    l_col = kmeans_col.n_clusters
    # label_lsit_1_row = kmeans_row.labels_
    label_lsit_1_col = kmeans_col.labels_

    # result_row = []
    # for i in range(l_row):
    #     result_row.append(np.where(label_lsit_1_row == i))

    result_col = []
    for i in range(l_col):
        result_col.append(np.where(label_lsit_1_col == i)[0].tolist())

    # print(result_col)
    return result_col

