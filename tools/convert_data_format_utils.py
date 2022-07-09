"""
进行各种NER数据格式的转换
"""


def convert_cluener_to_conll_example(sample):
    """ 
    将单个cluener数据样例转换为conll格式

    :param: sample (dict): cluener的一个数据样例
    :return: char_tag_list (list): conll格式的数据样例, [(char, tag), (char, tag), ...]
    """
    
    sentence = sample['text']
    
    hashmap = {}
    for entity_type in sample['label']:
        # print(sample['label'][entity_type])
        for entity in sample['label'][entity_type]:
            # print(entity)
            indexes = sample['label'][entity_type][entity]
            start_to_end_index = indexes[0]
            start, end = start_to_end_index[0], start_to_end_index[1]
            hashmap[start_to_end_index[0]] = "B-" + entity_type
            for i in range(start + 1, end + 1):
                hashmap[i] = "I-" + entity_type
    
    char_tag_list = []
    for i in range(len(sentence)):
        char = sentence[i]
        if i in hashmap:
            tag = hashmap[i]
        else:
            tag = "O"
        char_tag_list.append((char, tag))
    
    return char_tag_list  


def convert_conll_to_json_example(char_tag_list):
    """
    将单个conll格式的数据样例转换成JSON格式

    :param: char_tag_list (list): conll格式的数据样例, [(char, tag), (char, tag), ...]
    :return: json_dict (dict): json格式的数据样例 {'text': '***', 'labels': "tag1 tag2 tag3"}}
    
    """
    json_dict = {}
    char_list, tag_list = [], []
    for i in range(len(char_tag_list)):
        char, tag = char_tag_list[i]
        char_list.append(char)
        tag_list.append(tag)

    # 主要是处理中文的情况, 因此字符直接进行拼接
    # 标签之间需要用空格分隔
    json_dict['text'] = ''.join(char_list)
    json_dict['labels'] = tag_list
    
    return json_dict
    