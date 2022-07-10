"""
评估模型表现的一些工具函数
"""

def get_entity_bio(seq, id2label):
    """
    从标注序列中解析实体, 标注序列是以BIO的方式标注的
    
    :param seq (list): 标注序列, 其中每个元素是一个标签
    :param id2label (dict): id到label的映射字典
    :return (list): list of (chunk_type, chunk_start, chunk_end).
    Example:
        seq = ['B-PER', 'I-PER', 'O', 'B-LOC']
        get_entity_bio(seq)
        #output
        [['PER', 0,1], ['LOC', 3, 3]]
    """
    chunks = []
    chunk = [-1, -1, -1]
    for indx, tag in enumerate(seq):
        if not isinstance(tag, str):
            # 预测时, 一般得到的是标签索引, 这里将其转换为标签
            tag = id2label[tag]
        if tag.startswith("B-"):
            if chunk[2] != -1:
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


def get_entities(seq, id2label, markup='bio'):
    '''
    从标注序列中解析实体
    
    :param seq (list): 标注序列, 其中每个元素是一个标签
    :param id2label (dict): id到label的映射字典
    :param markup (str): 标注方式, 是'bio'还是'bioes'等
    :return (list): list of (chunk_type, chunk_start, chunk_end).
    '''
    assert markup in ['bio','bios']
    
    if markup =='bio':
        return get_entity_bio(seq, id2label)
