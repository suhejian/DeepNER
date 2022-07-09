"""
进行各种NER数据格式的转换
"""

import pandas as pd
import json
import tools.convert_data_format_utils as utils


def convert_cluener_to_conll_json(cluener_file_path, conll_file_path):
    """
    将CLUENER数据的原始格式转换成SequenceLabeling的格式

    :param: cluener_file_path (str): CLUENER数据的文件路径, 只能接受训练集和验证集路径, 因为测试集没有标签
    :conll_file_path (str): 转换后的数据存储文件路径
    """

    with open(cluener_file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    data = [json.loads(line) for line in lines]

    conll_data = {'text': [], 'labels': []}
    for sample in data:
        # 每个样本对应DataFrame的一行
        char_tag_list = utils.convert_cluener_to_conll_example(sample)
        json_dict = utils.convert_conll_to_json_example(char_tag_list)

        conll_data['text'].append(json_dict['text'])
        conll_data['labels'].append(json_dict['labels'])

    with open(conll_file_path, 'w', encoding='utf-8') as f:
        for text, labels in zip(conll_data['text'], conll_data['labels']):
            f.write(json.dumps({'text': text, 'labels': labels}, ensure_ascii=False) + '\n')

