#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@File    :   tab_post.py    
@Author  :   name
@Modify Time      @Version    @Desciption
------------      --------    -----------
2021/4/01 下午2:09    1.0         None
"""
import numpy as np
import json
import os

from common import iou, k_mean
from utils import format_html


def get_parent_dict(uf_path, df_path, lm_path, rm_path, img_path_txt, number_per_file=2):
    """
    Get parent_dict from the predict results of PointerNet.
    Args:
        uf_path: The results of up father
        df_path: The results of down father
        lm_path: The results of left mother
        rm_path: The results of right mother
        img_path_txt: The file list processed by PointerNet
        number_per_file: The count of results per file

    Returns:
        parent_dict:
            key: img_file_name
            value: parents list
                [
                    [up_father_list],
                    [down_father_list],
                    [left_mother_list],
                    [right_mother_list]
                ]
    """
    parent_dict = {}

    with open(img_path_txt, 'r') as jf:
        img_list = jf.readlines()

    for p_index in range(0, len(img_list), number_per_file):
        for parent_path in [uf_path, df_path, lm_path, rm_path]:
            parent_ptr = []
            if parent_path != '':
                parent_file = parent_path + str(int(p_index / number_per_file)) + '.txt'
                parent_ptr = np.loadtxt(open(parent_file, "rb"), delimiter=" ", skiprows=0)
            for n_index in range(number_per_file):
                img_file_name = img_list[p_index + n_index].strip().split('/')[-1]
                if img_file_name not in parent_dict:
                    parent_dict[img_file_name] = []
                try:
                    parent_dict[img_file_name].append([int(item) - 2 for item in parent_ptr[n_index]])
                except:
                    parent_dict[img_file_name].append([])

    # print(parent_dict)
    return parent_dict


def get_specific_files(file_path, exts=['.json']):
    """
    Get the .json file list from the predict results of OCR.
    Args:
        file_path: The results path of OCR
        exts: List of extension type

    Returns:
        file_list: List of ['.json'] file
    """
    file_list = []
    files = os.listdir(file_path)
    for file in files:
        (filename, extension) = os.path.splitext(file)
        if extension in exts:
            file_list.append(file)

    # print(len(file_list))
    return file_list


def parse_ocred_json(json_file):
    """
    Parse the json_file of OCR.
    Args:
        json_file: The json_file of OCR

    Returns:
        pred_row_list: Row list of cells
            [[[index, text, box],...],...]  # index starts from 1
    """
    pred_row_list = []
    cell_index = 0
    with open(json_file, 'rb') as jf:
        table_json = json.load(jf)
        # print(table_json)
        for row_index, row in enumerate(table_json['pages'][0]['rows']):
            row_list = []
            for col_index, col in enumerate(row):
                cell_index += 1
                pred_cell = [cell_index]
                pred_cell.append(row[col_index]['text'])
                pred_cell.append([row[col_index]['area']['top_left_x'],
                                  row[col_index]['area']['top_left_y'],
                                  row[col_index]['area']['bottom_right_x'],
                                  row[col_index]['area']['bottom_right_y']])
                row_list.append(pred_cell)
            pred_row_list.append(row_list)

    # print(pred_row_list)
    return pred_row_list


def merge_by_iou(preds):
    """
    Just process the adjacent lines.
    Args:
        preds: pred_row_list

    Returns:
        preds: Merge some rows according to "w_iou < 0.2 and h_iou > 0.45"
    """
    # col_width_max = max([len(row) for row in preds])
    total_cell_count = sum([len(row) for row in preds])
    i_list = []     # The index of rows merged into other rows

    def take_x_min(elem):
        return elem[2][0]

    for i, _ in enumerate(preds):
        if i == 0 or i-1 in i_list:
            continue
        _, w_iou, h_iou = iou(preds[i-1][0][2], preds[i][0][2])
        if w_iou < 0.2 and h_iou > 0.45:    # and len(preds[i-1]) + len(preds[i]) <= col_width_max:
            preds[i-1] += preds[i]
            preds[i-1] = sorted(preds[i-1], key=take_x_min)
            preds[i] = []
            i_list.append(i)

    for i in sorted(i_list, reverse=True):      # delete the row merged into other rows
        if len(preds[i]) == 0:
            del preds[i]

    cell_count = 0      # renumber the preds
    for pred in preds:
        for p in pred:
            cell_count += 1
            p[0] = cell_count
    assert total_cell_count == cell_count

    # col_width_max = max([len(row) for row in preds])
    # print(col_width_max)
    return preds


def preds_to_matrix(preds, parent_dict):
    """
    Recover the results of OCR and PointerNet to adjacency matrix.
    Args:
        preds: pred_row_list
        parent_dict:
            key: img_file_name
            value: parents list

    Returns:
        ret_list: The merged pred_list
        matrix: The adjacency matrix corresponding to HTML
    """
    row_length = [len(row) for row in preds]
    col_width_max = max(row_length)
    row_with_max_cols = [index for index, length in enumerate(row_length) if length == col_width_max]

    preds_list = []  # Flaten the preds
    for pred in preds:
        preds_list += pred

    if row_with_max_cols * col_width_max == sum(row_length):    # simple table
        ret_list = preds_list
        matrix = np.array(list(range(1, len(row_with_max_cols)*col_width_max+1))).reshape(len(row_with_max_cols), col_width_max)
    else:   # complex table
        matrix = complex_table_over_split(preds, row_length, col_width_max, row_with_max_cols)
        matrix = complex_matrix_row_merge(matrix, parent_dict[0])
        ret_list, matrix = complex_matrix_col_merge(matrix, preds_list, parent_dict[1], parent_dict[2])
    return ret_list, matrix


def find_col_index(cell, cols, refer):
    """
    Find the cell should be placed in which column.
    Args:
        cell: Cell to be placed
        cols: The columns clustered by KMeans
        refer: One row of over split matrix

    Returns:
        cell_index: The column index that the cell should be placed in
    """
    cell_index = None
    cell_col = 0
    for cl_index, col in enumerate(cols):
        if cell in col:
            cell_col = cl_index
            break
    for col_cell in cols[cell_col]:
        if col_cell in refer:
            cell_index = refer.index(col_cell)
            break
    return cell_index


def complex_table_over_split(preds, row_length, col_width_max, row_with_max_cols):
    """

    Args:
        preds: pred_row_list
        row_length: The length of every row in preds
        col_width_max: The max width of column in preds
        row_with_max_cols: The index of row with maximum width column

    Returns:
        matrix: The over split adjacency matrix to be merged in rows and columns
    """
    preds_list = []     # Flaten the preds
    for pred in preds:
        preds_list += pred

    cols = k_mean(preds_list, len(row_length), col_width_max)
    refer = [pred[0] - 1 for pred in preds[row_with_max_cols[0]]]

    matrix = np.zeros((len(row_length), col_width_max), dtype=int) - 1
    last_col = 0
    for r, row in enumerate(preds):
        for c, col in enumerate(row):
            ci = c
            if len(row) < col_width_max:    # Find the col index for short row
                ret = find_col_index(preds[r][c][0] - 1, cols, refer)
                if ret is None:
                    if last_col == col_width_max - 1:
                        ci = 0
                    else:
                        ci = last_col + 1
                else:
                    ci = ret
            matrix[r, ci] = preds[r][c][0]
            last_col = ci
    matrix[matrix < 0] = 0

    # print(matrix)
    return matrix


def complex_matrix_row_merge(matrix, uf_list, df_list=[], max_limit=50):
    """

    Args:
        matrix: The adjacency matrix to be merged in rows
        uf_list: The predicted up fathers by PointerNet
        df_list: The predicted down fathers by PointerNet
        max_limit: The max length of PointerNet result

    Returns:
        matrix: The adjacency matrix merged in rows
    """
    for i in range(matrix.shape[0]-1, -1, -1):
        for j in range(matrix.shape[1]-1, -1, -1):
            if matrix[i, j] == 0 or matrix[i, j] > max_limit:
                continue
            if matrix[i, j] in uf_list:
                colspan = uf_list.count(matrix[i, j])
                if colspan > 1:
                    where = np.where(matrix == uf_list.index(matrix[i, j])+1)
                    if len(where[1]) > 0:
                        child = where[1][0]
                        min_l = [t for t, cell in enumerate(matrix[i, :j]) if cell > 0]
                        cur_min = max(min_l) if len(min_l) > 0 else -1
                        max_l = [t for t, cell in enumerate(matrix[i, :]) if cell > matrix[i, j]]
                        cur_max = min(max_l) if len(max_l) > 0 else matrix.shape[1]
                        colspan = min([colspan, cur_max-cur_min-1])
                        index_min = max(child, cur_min+1)
                        index_max = min(child + colspan, cur_max)

                        matrix[i, index_min:index_max] = matrix[i, j]
    # print(matrix)
    return matrix


def complex_matrix_col_merge(matrix, preds_list,  lm_list, rm_list, max_limit=50):
    """

    Args:
        matrix: The adjacency matrix to be merged in columns
        preds_list: The preds_list to be merged in columns
        lm_list: The predicted left mothers by PointerNet
        rm_list: The predicted right mothers by PointerNet
        max_limit: The max length of PointerNet result

    Returns:
        preds_list: The preds_list merged in columns
        matrix: The adjacency matrix merged in columns
    """
    br = False
    for i in range(matrix.shape[0]):
        if br:
            break
        for j in range(matrix.shape[1]):
            if br:
                break
            if matrix[i, j] == 0:
                continue
            if matrix[i, j] > max_limit:
                br = True
                break

            m_list_l, m_list_r = set(), set()
            if matrix[i, j] in lm_list and lm_list.count(matrix[i, j]) > 1:
                m_list_l = set([index + 1 for index, cell in enumerate(lm_list) if cell == matrix[i, j]])
            if matrix[i, j] in rm_list and rm_list.count(matrix[i, j]) > 1:
                m_list_r = set([index + 1 for index, cell in enumerate(rm_list) if cell == matrix[i, j]])

            m_list = list(m_list_l & m_list_r)      # The intersection of left and right mothers
            m_list = sorted(m_list)
            for m_i, m_c in enumerate(m_list):
                if m_i == 0:
                    m_first = m_c
                else:
                    preds_list[m_first - 1][1] = preds_list[m_first - 1][1] + preds_list[m_c - 1][1]
                if m_c in matrix:
                    w_c = np.where(matrix == m_c)
                    for k in range(len(w_c[0])):
                        matrix[w_c[0][k], w_c[1][k]] = 0

    # print(matrix)
    ri_list = []
    for ri, row in enumerate(matrix):
        if sum(matrix[ri]) != 0:
            ri_list.append(ri)
        else:
            print("==========================")
    matrix = matrix[[ri_list], :][0]
    print(matrix)

    return preds_list, matrix


def matrix_to_html(label_list, adjacent_matrix):
    """
    Generate the json with HTML element according to label_list and adjacent_matrix
    Args:
        label_list: The merged pred_list
        adjacent_matrix: The merged final adjacency matrix corresponding to HTML

    Returns:
        img_json:
            "html": {
                "cells": cells_tokens,
                "structure": {
                    "tokens": structure_tokens
                }
            }
    """
    shape = adjacent_matrix.shape
    print(shape)
    # body_index = [l[3] for l in label_list].index(0)
    body_index = adjacent_matrix.shape[1]
    assert body_index != 0

    cells_tokens = []
    structure_tokens = ["<thead>"]
    for i in range(shape[0]):
        if i > 0 and max(adjacent_matrix[i-1, :]) == body_index:
            structure_tokens += ["</thead>", "<tbody>"]
        structure_tokens.append("<tr>")
        for j in range(shape[1]):
            if adjacent_matrix[i][j] == 0:
                cells_tokens.append({"tokens": []})
                structure_tokens += ["<td>", "</td>"]
            elif (j > 0 and adjacent_matrix[i][j] == adjacent_matrix[i][j-1]) \
                    or (i > 0 and adjacent_matrix[i][j] == adjacent_matrix[i-1][j]):
                continue
            else:
                cells_tokens.append({"tokens": [label_list[adjacent_matrix[i][j]-1][1]]})
                row_same = j < shape[1]-1 and adjacent_matrix[i][j] == adjacent_matrix[i][j+1]
                col_same = i < shape[0]-1 and adjacent_matrix[i][j] == adjacent_matrix[i+1][j]
                if row_same and col_same:
                    rowspan_str = " rowspan=\"{}\"".format(list(adjacent_matrix[:, j]).count(adjacent_matrix[i][j]))
                    colspan_str = " colspan=\"{}\"".format(list(adjacent_matrix[j, :]).count(adjacent_matrix[i][j]))
                    structure_tokens += ["<td", rowspan_str, colspan_str, ">", "</td>"]
                elif col_same:
                    rowspan_str = " rowspan=\"{}\"".format(list(adjacent_matrix[:, j]).count(adjacent_matrix[i][j]))
                    structure_tokens += ["<td", rowspan_str, ">", "</td>"]
                elif row_same:
                    colspan_str = " colspan=\"{}\"".format(list(adjacent_matrix[i, :]).count(adjacent_matrix[i][j]))
                    structure_tokens += ["<td", colspan_str, ">", "</td>"]
                else:
                    structure_tokens += ["<td>", "</td>"]
        structure_tokens.append("</tr>")
    structure_tokens.append("</tbody>")
    img_json = {
        "html": {
            "cells": cells_tokens,
            "structure": {
                "tokens": structure_tokens
            }
        }
    }
    return img_json


def ret_json_generate():
    """
    Process the results of OCR and PointerNet to recovery .html and generate .json
    Returns:
        Generate html.json for metirc.py to calculate similarity.
    """
    ocred_path = '/home/gita/Downloads/mini_result/mini_json/'

    img_path_txt = '/home/gita/Downloads/mini_result_50/mini_father.txt'
    uf_path = '/home/gita/Downloads/mini_result_50/father/'
    df_path = ''
    lm_path = '/home/gita/Downloads/mini_result_50/mother_p/'
    rm_path = '/home/gita/Downloads/mini_result_50/mother_n/'

    ret_html_path = './recovered_html/mini_50/'
    ret_json_file = './src/mini_pred_50.json'

    father_dict = get_parent_dict(uf_path, df_path, lm_path, rm_path, img_path_txt, number_per_file=2)
    ocred_files = get_specific_files(ocred_path)

    count = 0
    pred_json = {}
    for file_name in ocred_files:
        if file_name in ['PMC3707453_006_00.png.json',
                         'PMC6022086_007_00.png.json']:
            continue
        preds = parse_ocred_json(ocred_path + file_name)
        preds = merge_by_iou(preds)
        count += 1
        print(count, file_name)
        preds_list, matrix = preds_to_matrix(preds, father_dict[file_name])
        if matrix is None:
            continue
        img_json = matrix_to_html(preds_list, matrix)
        pred_html = format_html(img_json)
        pred_json[file_name[:-5]] = pred_html
        with open(ret_html_path + file_name + '.html', 'w', encoding='utf-8') as hf:
            hf.write(str(pred_html))
        # break
    with open(ret_json_file, 'w') as tf:
        tf.write(json.dumps(pred_json).strip())


if __name__ == '__main__':
    ret_json_generate()
