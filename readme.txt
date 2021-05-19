tab_pre.py：

作用：
主要是为了生成符合PointerNet所需要的json格式的label数据

流程：
1. 读取PubTabNet_2.0.0.jsonl
2. 调用os.path.isfile(ocred_file_name)函数根据图像文件名称查看是否有对应的OCR的json结果
3. 调用parse_ocred函数把对应的OCR的json结果按rows解析
4. 同时调用dict_to_matrix函数根据PubTabNet_2.0.0.jsonl里面的'cells'和'structure'构造邻接矩阵adjacent_matrix和初步的label_list
5. 根据图像文件名称把图像文件二值化
6. 循环步骤4的label_list，从步骤5中的二值化图像中截取box大小的图像进行横向投影（好像开始是判断OCR的box和label的box，忘记后面为啥采用投影方式来着）
7. 根据步骤6的的投影调用getHProjection函数进行切行，确定该单元格内有多少行
8. 根据步骤7切好的行，重新进行label中【序号、文本、坐标的调整】，以及生成原始单元格对应的切行后编号new_cnt_index_list
9. --调用merge_matrix函数把步骤4中的adjacent_matrix和步骤8中的new_cnt_index_list生成GNN需要的格式
10. 调用down_to_up等函数根据adjacent_matrix和new_cnt_index_list按照规则查找切分后的每个cell的正/负父/母亲节点编号
11. 调用label_father函数将步骤9的节点编号添加到最终的label_list中
12. 写入label.json文件

---------------------
tab_post.py：

作用：
主要是根据PointerNet结果和OCR结果生成xxx_pred.json文件

流程：
1. 调用get_parent_dict函数解析PointerNet结果文件生成每个文件的正/负父/母亲节点列表
2. 循环OCR结果文件夹的每个json结果，调用parse_ocred_json函数按rows解析
3. 调用merge_by_iou函数将解析后的结果检查是否需要两行组成一行，并生成初步的preds
4. 在preds_to_matrix函数里面：
4.1 首先根据步骤3的preds判断是否每行的列数相同
4.2 如果相同认为是simple table，直接生成对应邻接矩阵
4.3 如果不同认为是complex table：
4.3.1 调用complex_table_over_split函数利用步骤3的preds中的最大列数进行聚类生成过分割矩阵
4.3.2 根据步骤1中的正/负父亲节点列表，调用complex_matrix_row_merge查看是否存在横向合并单元格
4.3.3 根据步骤1中的正/负母亲节点列表，调用complex_matrix_col_merge查看是否存在纵向合并单元格
4.3.4 生成最终的preds_list和matrix
5. 调用matrix_to_html函数根据步骤4的preds_list和matrix生成原始数据中的dict格式img_json
6. 调用format_html函数将步骤5中的img_json生成对应的html
7. 将所有文件的html按照格式要求生成对应的xxx_pred.json文件

链接: https://pan.baidu.com/s/1qJ1fXO7RuoFiYDTHKVdsTA 提取码: gtsn 复制这段内容后打开百度网盘手机App，操作更方便哦
--来自百度网盘超级会员v4的分享
