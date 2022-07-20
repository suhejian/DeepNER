"""
对数据集进行一些分析
"""

from collections import Counter


def data_length_info(data):
    """
    分析数据集的长度信息

    :param data (list): 对于训练集和验证集来说每个样本有'text'和'labels', 而测试集只有'text'
    """
    lengths = []
    for sample in data:
        lengths.append(len(sample['text']))
    
    num_sentences = len(data)
    min_length = min(lengths)
    max_length = max(lengths)
    avg_length = sum(lengths) / len(lengths)
    # 数据集中有多少句子, 最小的句子长度, 最大的句子长度, 平均句子长度
    return num_sentences, min_length, max_length, avg_length


def data_entity_info(data):
    """
    分析数据集的实体信息

    :param data (list): 对于训练集和验证集来说每个样本有'text'和'labels', 而测试集只有'text'
    """
    num_entities = 0
    entity_counter = Counter()
    for sample in data:
        chunks = get_entities(seq=sample['labels'])
        num_entities += len(chunks)
        entity_counter.update(Counter([chunk[0] for chunk in chunks]))
    # 实体总数, 各实体类型的分布
    return num_entities, entity_counter

def get_entity_bio(seq):
    """
    由标签序列, 得到实体的相关信息
    标注序列是以BIO的方式标注的

    :param seq (list): 标注序列, 其中每个元素是一个标签
    :return (list): list of (chunk_type, chunk_start, chunk_end)
    Example:
        seq = ['B-PER', 'I-PER', 'O', 'B-LOC']
        get_entity_bio(seq)
        #output
        [['PER', 0,1], ['LOC', 3, 3]]    
    """
    chunks = []
    chunk = [-1, -1, -1]
    for indx, tag in enumerate(seq):
        if tag.startswith("B-"):
            if chunk[2] != -1:
                # 该chunk已经构成一个实体
                chunks.append(chunk)
            chunk = [-1, -1, -1]
            chunk[1] = indx
            chunk[0] = tag.split('-')[1]
            chunk[2] = indx
            if indx == len(seq) - 1:
                chunks.append(chunk)
        elif tag.startswith('I-') and chunk[1] != -1:
            _type = tag.split('-')[1]
            if _type == chunk[0]:
                chunk[2] = indx

            if indx == len(seq) - 1:
                chunks.append(chunk)
        else:
            if chunk[2] != -1:
                chunks.append(chunk)
            chunk = [-1, -1, -1]
    
    return chunks


def get_entities(seq, markup='bio'):
    '''
    从标注序列中解析实体
    
    :param seq (list): 标注序列, 其中每个元素是一个标签
    :param markup (str): 标注方式, 是'bio'还是'bioes'等
    :return (list): list of (chunk_type, chunk_start, chunk_end).
    '''
    assert markup in ['bio']
    
    if markup =='bio':
        return get_entity_bio(seq)