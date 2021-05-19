#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@File    :   tab_pre.py
@Author  :   name
@Modify Time      @Version    @Desciption
------------      --------    -----------
2021/4/23 下午7:08    1.0         None
"""

import cv2
import json
import jsonlines
import os.path
import re

import numpy as np


def dict_to_matrix(line_dict):
    """

    Args:
        line_dict: A line read from PubTabNet_2.0.0.jsonl

    Returns:
        label_list: [table_id, box, content, type_id]
        adjacent_matrix: The adjacent matrix of the cells in table.
        Examples:
            [[ 1  2  3  4  5  6  7  8]
             [ 0  0  9  9  9  9  9  9]
             [10 11 12 13 14 15 16 17]
             ...
             [82 83 84 85 86 87 88 89]
             [82 90 91 92 93 94 95 96]]

            # 0: The cell has nothing
            # 9  9  9  9  9  9: The colspan of cell is 6
            # 82  ...
              82  ... : The rowspan of cell is 2
    """
    structure = line_dict['html']['structure']['tokens']
    head_start = structure.index("<thead>")
    head_end = structure.index("</thead>")
    body_start = structure.index("<tbody>")
    body_end = structure.index("</tbody>")
    head = structure[head_start + 1:head_end]
    body = structure[body_start + 1:body_end]
    total_cells = head + body

    '''[begin] calculate row_count & col_count'''
    row_count = structure.count('</tr>')  # total rows
    first_row = head[head_start:head.index("</tr>")+1]
    # print(first_row)
    col_start, col_count = 0, 0
    for col_end in [index for (index, v) in enumerate(first_row) if v == '</td>']:
        col_str = ''.join(first_row[col_start+1:col_end+1])
        col_start = col_end
        # print(col_str)
        if 'colspan' in col_str:
            p = re.findall(re.compile(r'colspan=["](.*?)["]', re.S), col_str)
            assert len(p) == 1
            col_count += int(p[0])
        else:
            col_count += 1
    # print(row_count, col_count)
    '''[end] calculate row_count & col_count'''

    '''[begin] type_list (-1:head,0:body)'''
    head_cells_count = head.count('</td>')  # head cells
    body_cells_count = body.count('</td>')  # body cells
    type_list = [-1]*head_cells_count + [0]*body_cells_count  # [-1, -1, -1, ... 0, 0, 0]
    '''[end] type_list (-1:head,0:body)'''

    label_list = []  # [[1, [1, 4, 27, 13], '<b>Variable</b>', -1],...,[69, [336, 381, 376, 391], '0.214–1.651', 0], []]
    content_index_list = []  # [1, 2, 3, 4, 5, 0, 0, ..., 0, 67, 68, 69, 0]
    content_index = 0
    for index, item in enumerate(line_dict['html']['cells']):
        label = []
        if 'bbox' in item:
            content_index += 1
            label.append(content_index)
            label.append(''.join(item['tokens']))
            label.append(item['bbox'])
            label.append(type_list[index])
            label_list.append(label)
            content_index_list.append(content_index)
        else:
            content_index_list.append(0)
    # print(label_list)
    # print(content_index_list)
    # index_max = max(content_index_list)

    '''[begin] calculate adjacent_matrix'''
    head_rows_count = head.count('</tr>')  # head rows
    body_rows_count = body.count('</tr>')  # body rows
    adjacent_matrix = np.zeros((row_count, col_count), dtype=int) - 1
    row_start = 0
    content_index = 0
    for row_index, row_end in enumerate([index for (index, v) in enumerate(total_cells) if v == '</tr>']):
        row = total_cells[row_start:row_end+1]
        row_start = row_end + 1
        # print(row)
        col_count_in_row = row.count('</td>')
        assert col_count_in_row <= col_count
        col_index = 0
        col_start_in_row = 0
        for col_end_in_row in [index for (index, v) in enumerate(row) if v == '</td>']:
            col_str = ''.join(row[col_start_in_row+1:col_end_in_row+1])
            col_start_in_row = col_end_in_row
            # print(row_index, col_index)
            # print(adjacent_matrix)
            while adjacent_matrix[row_index, col_index] != -1:
                col_index += 1
            col_span = 1
            if 'colspan' in col_str or 'rowspan' in col_str:
                if 'colspan' in col_str and 'rowspan' in col_str:
                    p_col = re.findall(re.compile(r'colspan=["](.*?)["]', re.S), col_str)
                    p_row = re.findall(re.compile(r'rowspan=["](.*?)["]', re.S), col_str)
                    assert len(p_col) == 1
                    assert len(p_row) == 1
                    col_span = int(p_col[0])
                    assert col_span <= col_count
                    row_span = int(p_row[0])
                    if row_index < head_rows_count:
                        assert row_span <= head_rows_count
                    else:
                        assert row_span <= body_rows_count
                    adjacent_matrix[row_index:row_index+row_span, col_index:col_index+col_span] = content_index_list[content_index]
                elif 'colspan' in col_str:
                    p_col = re.findall(re.compile(r'colspan=["](.*?)["]', re.S), col_str)
                    assert len(p_col) == 1
                    col_span = int(p_col[0])
                    assert col_span <= col_count
                    adjacent_matrix[row_index, col_index:col_index+col_span] = content_index_list[content_index]

                elif 'rowspan' in col_str:
                    p_row = re.findall(re.compile(r'rowspan=["](.*?)["]', re.S), col_str)
                    assert len(p_row) == 1
                    row_span = int(p_row[0])
                    if row_index < head_rows_count:
                        assert row_span <= head_rows_count
                    else:
                        assert row_span <= body_rows_count
                    adjacent_matrix[row_index:row_index+row_span, col_index] = content_index_list[content_index]
            else:
                adjacent_matrix[row_index, col_index] = content_index_list[content_index]
            content_index += 1
            col_index += col_span
    '''[end] calculate adjacent_matrix'''
    # print(label_list)
    # print(adjacent_matrix)
    assert -1 not in adjacent_matrix
    return label_list, adjacent_matrix


def down_to_up(adjacent_matrix):
    """
    Find father from down to up.
    Args:
        adjacent_matrix: The adjacent matrix of the cells in table.

    Returns:
        father_list: The fathers of cell which has content.
    """
    father_list = []
    max_index = max(max(row) for row in adjacent_matrix)
    for s in range(max_index):
        father_list.append(set())

    shape = adjacent_matrix.shape
    for i in range(shape[0]):
        for j in range(shape[1]):
            if adjacent_matrix[i, j] == 0:
                continue
            if (i == 0 and j > 0 and adjacent_matrix[i, j] == adjacent_matrix[i, j-1]) or \
                    (i > 0 and adjacent_matrix[i, j] == adjacent_matrix[i - 1, j]):
                continue
            up_trace = -1
            if i == 0:
                father = -1
            else:
                up_trace = i - 1
                while up_trace >= 0 and adjacent_matrix[up_trace, j] == 0:
                    up_trace -= 1
                if up_trace == -1:
                    father = -1
                else:
                    father = adjacent_matrix[up_trace, j]
            father_list[adjacent_matrix[i, j] - 1].add((i-up_trace, father))
            # assert len(father_list) <= adjacent_matrix[i, j]
    # print(father_list)
    return father_list


def up_to_down(adjacent_matrix):
    """
    Find father from up to down.
    Args:
        adjacent_matrix: The adjacent matrix of the cells in table.

    Returns:
        father_list: The fathers of cell which has content.
    """
    father_list = []
    max_index = max(max(row) for row in adjacent_matrix)
    for s in range(max_index):
        father_list.append(set())

    shape = adjacent_matrix.shape
    for i in range(shape[0]):
        for j in range(shape[1]):
            if adjacent_matrix[i, j] == 0:
                continue
            if (i == shape[0]-1 and j > 0 and adjacent_matrix[i, j] == adjacent_matrix[i, j-1]) or \
                    (i > 0 and adjacent_matrix[i, j] == adjacent_matrix[i - 1, j]):
                continue
            up_trace = shape[0]
            if i == shape[0]-1:
                father = -1
            else:
                up_trace = i + 1
                while up_trace <= shape[0]-1 and (adjacent_matrix[up_trace, j] == 0 or
                                                  adjacent_matrix[up_trace, j] == adjacent_matrix[i, j]):
                    up_trace += 1
                if up_trace == shape[0]:
                    father = -1
                else:
                    father = adjacent_matrix[up_trace, j]
            father_list[adjacent_matrix[i, j] - 1].add((up_trace - i , father))
    # print(father_list)
    return father_list


def right_to_left(adjacent_matrix):
    """
    Find father from right to left.
    Args:
        adjacent_matrix: The adjacent matrix of the cells in table.

    Returns:
        father_list: The fathers of cell which has content.
    """
    father_list = []
    max_index = max(max(row) for row in adjacent_matrix)
    for s in range(max_index):
        father_list.append(set())

    shape = adjacent_matrix.shape
    for j in range(shape[1]):
        for i in range(shape[0]):
            if adjacent_matrix[i, j] == 0:
                continue
            if (j == 0 and i > 0 and adjacent_matrix[i, j] == adjacent_matrix[i-1, j]) or \
                    (j > 0 and adjacent_matrix[i, j] == adjacent_matrix[i, j-1]):
                continue
            up_trace = -1
            if j == 0:
                father = -1
            else:
                up_trace = j - 1
                while up_trace >= 0 and adjacent_matrix[i, up_trace] == 0:
                    up_trace -= 1
                if up_trace == -1:
                    father = -1
                else:
                    father = adjacent_matrix[i, up_trace]
            father_list[adjacent_matrix[i, j] - 1].add((j-up_trace, father))

    # print(father_list)
    return father_list


def left_to_right(adjacent_matrix):
    """
    Find father from left to right.
    Args:
        adjacent_matrix: The adjacent matrix of the cells in table.

    Returns:
        father_list: The fathers of cell which has content.
    """
    father_list = []
    max_index = max(max(row) for row in adjacent_matrix)
    for s in range(max_index):
        father_list.append(set())

    shape = adjacent_matrix.shape
    for j in range(shape[1]):
        for i in range(shape[0]):
            if adjacent_matrix[i, j] == 0:
                continue
            if (j == shape[1]-1 and i > 0 and adjacent_matrix[i, j] == adjacent_matrix[i-1, j]) or \
                    (j > 0 and adjacent_matrix[i, j] == adjacent_matrix[i, j-1]):
                continue
            up_trace = shape[1]
            if j == shape[1]-1:
                father = -1
            else:
                up_trace = j + 1
                while up_trace <= shape[1]-1 and (adjacent_matrix[i, up_trace] == 0 or
                                                  adjacent_matrix[i, up_trace] == adjacent_matrix[i, j]):
                    up_trace += 1
                if up_trace == shape[1]:
                    father = -1
                else:
                    father = adjacent_matrix[i, up_trace]
            father_list[adjacent_matrix[i, j] - 1].add((up_trace - j , father))
    # print(father_list)
    return father_list


def label_father(new_rect_list, father_list, index_list, father_type):
    """

    Args:
        new_rect_list: New boxes after the multi-line cell is split without parents.
        father_list: The father of element in adjacent matrix
        index_list: The element index_list is the number of lines which are merged in original table.
        father_type: One of ['up_father','down_father','left_mother','right_mother']

    Returns:
        new_rect_list: New boxes after the multi-line cell is split with parents.
    """
    for i, father in enumerate(father_list):
        min_index = min([i[0] for i in father])
        min_father = [int(item[1]) for item in father if item[0] == min_index]
        if father_type == 'up_father':
            for j, rect in enumerate(index_list[i]):
                if j == 0:
                    ret_father = max(index_list[min(min_father) - 1]) if min_father != [-1] else -1
                    new_rect_list[index_list[i][j]-1].append(ret_father)
                else:
                    ret_father = index_list[i][j-1]
                    new_rect_list[index_list[i][j]-1].append(ret_father)

        elif father_type == 'down_father':
            for j, rect in enumerate(index_list[i]):
                if j == len(index_list[i])-1:
                    ret_father = min(index_list[min(min_father) - 1]) if min_father != [-1] else -1
                    new_rect_list[index_list[i][j]-1].append(ret_father)
                else:
                    ret_father = index_list[i][j+1]
                    new_rect_list[index_list[i][j]-1].append(ret_father)

        elif father_type == 'left_mother' or father_type == 'right_mother':
            ret_father = min(index_list[min(min_father) - 1]) if min_father != [-1] else -1
            for j, rect in enumerate(index_list[i]):
                new_rect_list[index_list[i][j]-1].append(ret_father)

    return new_rect_list


'''---------------------------------------- Begin Label Create ----------------------------------------'''


def parse_ocred(ocred_file):
    cell_index = 0
    preds = []
    with open(ocred_file, 'rb') as jf:
        table_json = json.load(jf)
        # print(table_json)
        for row_index, row in enumerate(table_json['pages'][0]['rows']):
            for col_index, col in enumerate(row):
                cell_index += 1
                pred_cell = [cell_index]
                pred_cell.append(row[col_index]['text'])
                pred_cell.append([row[col_index]['area']['top_left_x'],
                                 row[col_index]['area']['top_left_y'],
                                 row[col_index]['area']['bottom_right_x'],
                                 row[col_index]['area']['bottom_right_y']])
                preds.append(pred_cell)
        # print(preds)
        return preds


def merge_matrix(file_name, adjacent_matrix, new_cnt_index_list):
    """
    This function is used to generate matrix for GNN.
    Args:
        file_name:
        adjacent_matrix:
        new_cnt_index_list:

    Returns:
        ret_matrix:

    """
    ret_matrix = 0
    shape = adjacent_matrix.shape
    for i in range(shape[0]):
        index_list = [index for index in adjacent_matrix[i]]
        len_list = []
        for index in adjacent_matrix[i]:
            if index == 0:
                len_list.append(-1)
            else:
                len_list.append(len(new_cnt_index_list[index - 1]))
        row_expand = max(len_list)

        if row_expand == -1:
            sub_matrix = np.zeros((1, shape[1]), dtype=int)
            if i == 0:
                ret_matrix = sub_matrix
            else:
                ret_matrix = np.vstack((ret_matrix, sub_matrix))
            continue

        sub_matrix = np.zeros((row_expand, shape[1]), dtype=int)
        for k in range(shape[1]):
            if len_list[k] == -1:
                continue
            for j in range(len_list[k]):
                sub_matrix[j][k] = new_cnt_index_list[index_list[k]-1][j]  # if index_list[k] != 0 else 0
        if i == 0:
            ret_matrix = sub_matrix
        else:
            ret_matrix = np.vstack((ret_matrix, sub_matrix))

    # print(ret_matrix)
    np.savetxt('/home/gita/Documents/BACK/pubtab/json_40_label/matrix/' + file_name + '.txt', ret_matrix)


def getHProjection(image):
    hProjection = np.zeros(image.shape, np.uint8)
    # 图像高与宽
    (h, w) = image.shape
    # 长度与图像高度一致的数组
    h_ = [0] * h
    # 循环统计每一行白色像素的个数
    for y in range(h):
        for x in range(w):
            if image[y, x] == 255:
                h_[y] += 1
    # 绘制水平投影图像
    # for y in range(h):
    #     for x in range(h_[y]):
    #         hProjection[y, x] = 255
    # cv2.imshow('hProjection2', hProjection)

    return h_


def getVProjection(image):
    vProjection = np.zeros(image.shape, np.uint8);
    # 图像高与宽
    (h, w) = image.shape
    # 长度与图像宽度一致的数组
    w_ = [0] * w
    # 循环统计每一列白色像素的个数
    for x in range(w):
        for y in range(h):
            if image[y, x] == 255:
                w_[x] += 1
    # 绘制垂直平投影图像
    for x in range(w):
        for y in range(h - w_[x], h):
            vProjection[y, x] = 255
    # cv2.imshow('vProjection',vProjection)
    return w_


def split_line(datafile, filters=[]):
    """
    Generate the label json file.
    Args:
        datafile: The original jsonl file
        filters: Some files to be ignored

    Returns:
        format :
        [[1, 'Name', [223, 17, 336, 48], 0, 0, -1, 4, -1, 2], ...]
    """
    count = 0
    err_count = 0
    exc_count = 0
    err_file_list = []
    ocred_file_path = '/home/gita/Downloads/mini_result/mini_json/'
    ori_image_path = './pubtabnet/val/'
    label_path = './val_mini_label/'
    with open(datafile, 'rb') as f:
        for line in f:
            line_dict = json.loads(line)
            file_name = line_dict['filename']
            if file_name in ['PMC3707453_006_00.png']:
                # if file_name not in filters:
                continue
            ocred_file_name = ocred_file_path + file_name + '.json'
            if not os.path.isfile(ocred_file_name):
                continue
            print('----------------------------------' + file_name)
            try:
                preds = parse_ocred(ocred_file_name)
                label_list, adjacent_matrix = dict_to_matrix(line_dict)
            except:
                exc_count += 1
                continue
            split_cls = line_dict['split']
            if split_cls == 'train':
                continue

            ori_image = cv2.imread(ori_image_path + file_name)
            gray_image = cv2.cvtColor(ori_image, cv2.COLOR_BGR2GRAY)
            bit_img = cv2.adaptiveThreshold(gray_image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 3,
                                            12)
            # bit_img = cv2.adaptiveThreshold(gray_image, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 3, 5)
            # _, bit_img = cv2.threshold(gray_image, 200, 255, cv2.THRESH_BINARY_INV)
            # cv2.imshow("bit_img", bit_img)
            # cv2.waitKey(0)
            new_rect_list = []
            new_cnt_index = 0
            new_cnt_index_list = []
            go_out = False
            for label in label_list:
                if go_out:
                    break
                s_list = [label[3]]
                for si, sc in enumerate(['</b>', '</i>', '</sup>', '</underline>']):
                    if sc in label[1]:
                        s_list.append(1)
                    else:
                        s_list.append(0)

                rect = label[2]
                if len(label[1]) == 1 or label[1] == '...':
                    new_cnt_index += 1
                    new_cnt_index_list.append([new_cnt_index])
                    new_label = [new_cnt_index] + label[1:] + [0]
                    new_rect_list.append(new_label)
                    continue
                rect_image = bit_img[rect[1]:rect[3], rect[0] + 1:rect[2] - 1]
                H = getHProjection(rect_image)
                start = 0
                H_Start = []
                H_End = []
                for i in range(len(H)):
                    if H[i] > 0 and start == 0:
                        H_Start.append(i)
                        start = 1
                    if H[i] <= 0 and start == 1:
                        H_End.append(i)
                        start = 0
                if len(H_Start) == 1:
                    new_cnt_index += 1
                    new_cnt_index_list.append([new_cnt_index])
                    new_label = [new_cnt_index] + label[1:] + [0]
                    new_rect_list.append(new_label)
                else:
                    if len(H_End) == len(H_Start) - 1:
                        H_End.append(rect[3] - rect[1])
                    assert len(H_End) == len(H_Start)
                    # print(label)
                    # print(H_Start)
                    # print(H_End)
                    del_list = []
                    for i in range(len(H_End)):
                        if H_End[i] - H_Start[i] <= 2:
                            del_list.append(i)
                    del_list.sort(reverse=True)
                    for i in range(len(del_list)):
                        del H_End[del_list[i]]
                        del H_Start[del_list[i]]
                    # print(H_Start)
                    # print(H_End)
                    if len(H_Start) == 0:
                        err_count += 1
                        print('*********************************************************')
                        err_file_list.append(file_name)
                        go_out = True
                        break
                    H_Start[0] = 0
                    H_End[-1] = rect[3] - rect[1]
                    for i in range(1, len(H_Start)):
                        H_Start[i] = H_End[i - 1] = int((H_Start[i] + H_End[i - 1]) / 2)
                    cell_cnt_index = []
                    for i in range(len(H_End)):
                        up = [0] if i == 0 else [new_cnt_index]
                        new_label = [[rect[0], rect[1] + H_Start[i], rect[2], rect[1] + H_End[i]]] + \
                                    [label[3]] + up
                        for p in preds:
                            if rect[0] < (p[2][0] + p[2][2]) / 2 < rect[2] and \
                                    rect[1] + H_Start[i] < (p[2][1] + p[2][3]) / 2 < rect[1] + H_End[i]:
                                new_label.insert(0, p[1])
                                break
                        if len(new_label) == 4:
                            new_cnt_index += 1
                            cell_cnt_index.append(new_cnt_index)
                            new_label.insert(0, new_cnt_index)

                            new_rect_list.append(new_label)
                        else:
                            go_out = True
                            break
                            err_file_list.append(file_name + "***")
                    new_cnt_index_list.append(cell_cnt_index)
                    # break
            if go_out:
                continue
            merge_matrix(file_name, adjacent_matrix, new_cnt_index_list)
            father_down_to_up = down_to_up(adjacent_matrix)
            father_up_to_down = up_to_down(adjacent_matrix)
            father_right_to_left = right_to_left(adjacent_matrix)
            father_left_to_right = left_to_right(adjacent_matrix)

            new_rect_list = label_father(new_rect_list, father_down_to_up, new_cnt_index_list, 'up_father')
            new_rect_list = label_father(new_rect_list, father_up_to_down, new_cnt_index_list, 'down_father')
            new_rect_list = label_father(new_rect_list, father_right_to_left, new_cnt_index_list, 'left_mother')
            new_rect_list = label_father(new_rect_list, father_left_to_right, new_cnt_index_list, 'right_mother')

            # print(new_rect_list)
            # print(new_cnt_index_list)

            # img = cv2.imread('./pubtabnet/train/' + file_name)
            # for rect in rect_list:
            #     cv2.rectangle(img, (rect[0], rect[1]), (rect[2], rect[3]), color=(0, 0, 255), thickness=1)
            # for rect_pred in preds:
            #     cv2.rectangle(img, (rect_pred[2][0], rect_pred[2][1]), (rect_pred[2][2], rect_pred[2][3]), color=(0, 255, 0), thickness=1)
            # for new_rect in new_rect_list:
            #     rect = new_rect[1]
            #     cv2.rectangle(img, (rect[0], rect[1]), (rect[2], rect[3]), color=(255, 0, 0), thickness=1)
            # cv2.imwrite('./test/' + file_name, img)
            
            with jsonlines.open(label_path + file_name + '.json', 'w') as jf:
                jf.write(new_rect_list)
            count += 1
            print(count)
            print(err_count, exc_count)
            # if count >= 100:
            #     break
            if count >= 20 - err_count - exc_count:
                print(err_count, exc_count)
                break

    with open('./err_file_list_mini.txt', 'w') as ef:
        ef.write(json.dumps(err_file_list))


if __name__ == '__main__':
    split_line('pubtabnet/PubTabNet_2.0.0.jsonl')
