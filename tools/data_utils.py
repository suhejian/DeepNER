"""
做一些数据预处理工作
"""

import json
import os


def read_data(data_dir):
    """
    读取数据集

    :param data_dir (str): 数据集目录
    :return train_data (dict): 训练数据，包含输入文本和对应的标签序列
    :return test_data (dict): 测试数据，包含输入文本和对应的标签序列
    """

    assert os.path.exists(data_dir) == True, "数据集目录不存在"

    train_data = read_jsonl_file(os.path.join(data_dir + "/train.jsonl"))
    dev_data = read_jsonl_file(os.path.join(data_dir + "/dev.jsonl"))
    test_data = read_jsonl_file(os.path.join(data_dir + "/test.jsonl"))

    return train_data, dev_data, test_data


def read_jsonl_file(file_path):
    """
    读取jsonl文件
    
    :param file_path(str): jsonl文件路径
    :return data_list(list): jsonl文件中的数据
    """

    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    data_list = [json.loads(line) for line in lines]

    return data_list


def align_label_example(token_list, tag_list, tokenizer):
    """
    将标签与tokenized_input对齐
    
    :param token_list: 输入的句子, 以token列表的形式表示
    :param tag_list: 该句子对应的标签, 以tag列表的形式表示
    :param tokenizer: 用于处理输入句子的tokenizer
    :return tokenized_token_list: tokenizer处理以后的句子, 以token列表的形式表示
    :return aligned_tag_list: 对齐以后的标签, 以tag列表的形式表示
    """ 
    
    tokenized_token_list, aligned_tag_list = [], []

    assert len(token_list) == len(tag_list), "token_list和tag_list长度不一致"

    for token, tag in zip(token_list, tag_list):
        # tokenize the word and count the number of subwords
        tokenized_token = tokenizer.tokenize(token)
        n_subwords = len(tokenized_token)

        tokenized_token_list.extend(tokenized_token)
        aligned_tag_list.extend([tag] * n_subwords)

    # if the number of subwords is greater than the maximum length,
    # we need to truncate the tokenized_token_list
    assert len(tokenized_token_list) == len(aligned_tag_list), "tokenized_token_list和aligned_tag_list长度不一致"
    return tokenized_token_list, aligned_tag_list


def get_unique_tags(train_data):
    """
    获取所有标签的集合
    
    :param train_data (dict): 训练数据，包含输入文本和对应的标签序列
    # {"text": "生生不息CSOL生化狂潮让你填弹狂扫", \
    # "labels": ["O", "O", "O", "O", "B-game", "I-game", "I-game", "I-game", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O"]}
    :return unique_tags(list): 所有标签的集合
    :return labels_to_ids (dict): 标签到id的映射
    :return id_to_labels (dict): id到标签的映射
    """

    unique_tags = set()
    for sample in train_data:
        unique_tags.update(set(sample['labels']))

    unique_tags = sorted(list(unique_tags))

    labels_to_ids = {label: i for i, label in enumerate(unique_tags)}
    id_to_labels = {i: label for i, label in enumerate(unique_tags)}

    return unique_tags, labels_to_ids, id_to_labels
    

    