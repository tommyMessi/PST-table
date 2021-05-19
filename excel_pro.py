#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@File    :   chi_label.py    
@Author  :   name
@Modify Time      @Version    @Desciption
------------      --------    -----------
2021/4/13 下午4:46    1.0         None
"""
import json
import jsonlines

import numpy as np
import pandas as pd

from common import iou
from tab_pre import up_to_down, down_to_up, right_to_left, left_to_right, label_father
from tab_post import matrix_to_html, get_specific_files, get_parent_dict, preds_to_matrix
from utils import format_html


def parse_excel(file_name):
    """

    Args:
        file_name: Original excel files

    Returns:
        adjacent_matrix:
            [[ 1  2  3]     # The number is the order of cell which is not nan.
             [ 4  5  0]
             [ 6  7  8]
             [ 9 10 11]
             [12 13  0]
             [14 15 16]
             [17 18 19]]
        cnt_index_list:
            # The element is the cell content in original excel file.
            [[1], [2], [3], [4], [5], [8], [6, 9, 11], [7, 10, 12], [14], [15], [13, 16], [17], [18], [21], [19, 22], [20, 23], [24], [25], [26]]
    """
    df = pd.read_excel(file_name, header=None, engine='openpyxl')
    nd = np.array(df.values)
    print(nd)
    cnt_index_list = []
    cell_index = 0
    adjacent_matrix = np.zeros(nd.shape, dtype=int)  # - 1
    for i in range(nd.shape[0]):
        for j in range(nd.shape[1]):
            if not isinstance(nd[i, j], str) and np.isnan(nd[i, j]):
                continue
            if j > 0 and nd[i, j] == nd[i, j-1]:
                adjacent_matrix[i, j] = adjacent_matrix[i, j-1]
            elif i > 0 and nd[i, j] == nd[i-1, j]:
                adjacent_matrix[i, j] = adjacent_matrix[i-1, j]
            else:
                cell_index += 1
                adjacent_matrix[i, j] = cell_index
                if isinstance(nd[i, j], str):   # The number starts from 1
                    cnt_index = [int(s) + 1 for s in nd[i, j].replace('，', ',').split(',')]
                else:
                    cnt_index = [int(nd[i, j]) + 1]
                cnt_index_list.append(cnt_index)

    print(adjacent_matrix)
    print(cnt_index_list)
    return adjacent_matrix, cnt_index_list


def parse_json(file_name):
    """

    Args:
        file_name: Original json files

    Returns:
        label_list: label_list[1] is the original text which is not merged.
            [[1, '电池企业', [223, 17, 336, 48], 0],
             [2, '宁德时代', [771, 17, 883, 48], 0], ...]
    """
    label_list = []
    with open(file_name, 'r') as jf:
        for jd in json.load(jf):
            label = [jd['id'] + 1]
            label.append(jd['text'])
            label.append(jd['box'])
            label.append(0)
            label_list.append(label)
    return label_list


def excel_to_matrix(excel_file, text_file):
    """
    This adjacent_matrix is used to generate gt_json_file for calculating similarity.
    Args:
        excel_file: Original excel files
        text_file: Original json files

    Returns:
        adjacent_matrix:
            [[ 1  2  3]     # The number is the order of cell which is not nan.
             [ 4  5  0]
             [ 6  7  8]
             [ 9 10 11]
             [12 13  0]
             [14 15 16]
             [17 18 19]]
        cell_text_list: cell_text_list[1] is the merged text.
            [[1, '电池企业'],
             [2, '宁德时代'], ...]
    """
    adjacent_matrix, cnt_index_list = parse_excel(excel_file)
    ori_label_list = parse_json(text_file)
    cell_text_list = [[]] * max(max(row) for row in adjacent_matrix)

    for i in range(adjacent_matrix.shape[0]):
        for j in range(adjacent_matrix.shape[1]):
            ad_index = adjacent_matrix[i, j]
            cell_text = [ad_index]
            cell_text.append(''.join([ori_label_list[index-1][1] for index in cnt_index_list[ad_index - 1]]))
            cell_text_list[adjacent_matrix[i, j] - 1] = cell_text
    print(cell_text_list)
    return adjacent_matrix, cell_text_list


def excel_label_generate():
    """
    Generate the label json file.
    Returns:
        format :
        [[1, '电池企业', [223, 17, 336, 48], 0, 0, -1, 4, -1, 2], ...]
    """
    count = 0
    excel_path = '/home/gita/Downloads/rr_table/excel/'
    text_path = '/home/gita/Downloads/rr_table/text/'
    label_path = '/home/gita/Downloads/rr_table/label/'

    file_list = get_specific_files(excel_path, ['.xlsx'])
    for file in file_list:
        file_name = file.split('/')[-1].split('.')[0]
        adjacent_matrix, cnt_index_list = parse_excel(excel_path + file)
        label_list = parse_json(text_path + file_name + '.json')
        for cnt_index in cnt_index_list:
            if len(cnt_index) == 1:
                label_list[cnt_index[0] - 1].append(0)
            else:
                for ci, cnt_i in enumerate(cnt_index):
                    if ci == 0:
                        label_list[cnt_i - 1].append(0)
                    else:
                        label_list[cnt_i - 1].append(cnt_index[ci - 1])
        # print(label_list)
        new_rect_list = label_list
        father_down_to_up = down_to_up(adjacent_matrix)
        father_up_to_down = up_to_down(adjacent_matrix)
        father_right_to_left = right_to_left(adjacent_matrix)
        father_left_to_right = left_to_right(adjacent_matrix)

        new_rect_list = label_father(new_rect_list, father_down_to_up, cnt_index_list, 'up_father')
        new_rect_list = label_father(new_rect_list, father_up_to_down, cnt_index_list, 'down_father')
        new_rect_list = label_father(new_rect_list, father_right_to_left, cnt_index_list, 'left_mother')
        new_rect_list = label_father(new_rect_list, father_left_to_right, cnt_index_list, 'right_mother')

        with jsonlines.open(label_path + file_name + '.json', 'w') as jf:
            jf.write(new_rect_list)

        print(new_rect_list)
        count += 1
        print(count)
        # if count >= 1:
        #     break


def excel_gt_json_generate():
    """
    This function is used to generate rr_gt.json file for calculating similarity.
    Returns:
        Generate rr_gt.json for metirc.py to calculate similarity
    """
    count = 0
    gt_gt = {}
    excel_path = '/home/gita/Downloads/rr_table/excel/'
    text_path = '/home/gita/Downloads/rr_table/text/'
    html_path = '/home/gita/Downloads/rr_table/html/'
    father_path_file = '/home/gita/Downloads/final/f.txt'

    gt_json_file = './src/rr_gt.json'

    with open(father_path_file, 'r') as jf:
        file_list = jf.readlines()
    for file in file_list:
        file_name = file.split('/')[-1].split('.')[0]
        excel_file = excel_path + file_name + '.xlsx'
        text_file = text_path + file_name + '.json'
        adjacent_matrix, cell_text_list = excel_to_matrix(excel_file, text_file)
        gt_json = matrix_to_html(cell_text_list, adjacent_matrix)
        gt_html = format_html(gt_json)
        with open(html_path + file_name + '.html', 'w', encoding='utf-8') as hf:
            hf.write(str(gt_html))
        gt_gt[file_name + '.json'] = {
            "html": gt_html
        }
        count += 1
        print(count, file_name)
        # if count >= 1:
        #     break
    with open(gt_json_file, 'w') as tf:
        tf.write(json.dumps(gt_gt))


def merge_by_iou_v2(preds):
    """
    Merge rows which has only one original cell according to "w_iou < 0.2 and h_iou > 0.45"
    Args:
        preds: The original one cell per row

    Returns:
        preds: Merge some rows according to "w_iou < 0.2 and h_iou > 0.45"
    """
    print(preds)
    total_cell_count = sum([len(row) for row in preds])
    i_list = []

    def take_x_min(elem):
        return elem[2][0]

    for i, _ in enumerate(preds):
        if i in i_list:
            continue
        i_tmp = i + 1
        while i_tmp < total_cell_count:
            print(i_tmp, total_cell_count)
            _, w_iou, h_iou = iou(preds[i][0][2], preds[i_tmp][0][2])
            if w_iou < 0.2 and h_iou > 0.45: # and len(preds[i-1]) + len(preds[i]) <= col_width_max:
                preds[i] += preds[i_tmp]
                preds[i] = sorted(preds[i], key=take_x_min)
                preds[i_tmp] = []
                i_list.append(i_tmp)
            else:
                break
            i_tmp += 1

    for i in sorted(i_list, reverse=True):
        if len(preds[i]) == 0:
            del preds[i]

    print(preds)
    col_width_max = max([len(row) for row in preds])
    print(col_width_max)
    return preds


def axis_to_rows(file_name):
    """

    Args:
        file_name: The original json file

    Returns:
        preds: Merge some rows according to "w_iou < 0.2 and h_iou > 0.45"
    """
    label_list = parse_json(file_name)
    label_list.sort(key=lambda x: (int(x[2][1]), int(x[2][0])))
    label_list = [[label] for label in label_list]
    preds = merge_by_iou_v2(label_list)

    p_indexs = []
    for pred in preds:
        for p in pred:
            p_indexs.append(p[0]-1)
    print(p_indexs)
    # assert p_indexs == list(range(len(p_indexs)))

    print(preds)
    return preds


def excel_pred_json_generate():
    """
    Process the results of OCR and PointerNet to recovery .html and generate .json
    Returns:
        Generate rr_pred.json for metirc.py to calculate similarity.

    """
    count = 0
    error_count = 0
    text_path = '/home/gita/Downloads/rr_table/text/'

    uf_path = '/home/gita/Downloads/final/f1/'
    df_path = '/home/gita/Downloads/final/f2/'
    lm_path = '/home/gita/Downloads/final/m1/'
    rm_path = '/home/gita/Downloads/final/m2/'
    img_path_txt = '/home/gita/Downloads/final/f.txt'

    pred_html_path = './recovered_html/rr_html/'
    pred_json_file = './src/rr_pred.json'

    father_dict = get_parent_dict(uf_path, df_path, lm_path, rm_path, img_path_txt, number_per_file=2)
    file_list = get_specific_files('/home/gita/Downloads/rr_table/text/', ['.json'])
    pred_json = {}
    for file_name in file_list:
        # check_list = ['000917.json']
        # if file_name not in check_list:
        #     continue
        print(count, file_name)
        text_file = text_path + file_name
        # try:
        preds = axis_to_rows(text_file)
        # except:
        #     error_count += 1
        #     print("-------------------------------------------------------------")
        preds_list, matrix = preds_to_matrix(preds, father_dict[file_name])
        if matrix is None:
            continue
        img_json = matrix_to_html(preds_list, matrix)
        pred_html = format_html(img_json)

        with open(pred_html_path + file_name + '.html', 'w', encoding='utf-8') as hf:
            hf.write(str(pred_html))

        pred_json[file_name] = pred_html
        count += 1
        # if count >= len(check_list):
        #     break
    print(error_count)
    with open(pred_json_file, 'w') as tf:
        tf.write(json.dumps(pred_json).strip())


if __name__ == '__main__':
    # excel_label_generate()
    # excel_gt_json_generate()
    excel_pred_json_generate()

