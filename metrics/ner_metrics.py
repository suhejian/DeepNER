import torch
from collections import Counter
from metrics.metric_utils import get_entities

class SequenceLabelingEntityScore(object):
    def __init__(self, id2label, markup="BIO"):

        self.id2label = id2label
        self.markup = markup
        self.reset()

    def reset(self):
        self.origins = []   # ground truth
        self.founds = []    # predict
        self.rights = []    # right predict

    def compute(self, origin, found, right):
        # 这里的origin是原始的实体数量, found是预测的实体数量, right是预测正确的实体数量
        recall = 0 if origin == 0 else (right / origin)
        precision = 0 if found == 0 else (right / found)
        f1 = 0. if recall + precision == 0 else (2 * precision * recall) / (precision + recall)
        return recall, precision, f1

    def result(self):
        class_info = {}
        # x: [实体类型, 实体起始位置, 实体结束位置]
        origin_counter = Counter([x[0] for x in self.origins])
        found_counter = Counter([x[0] for x in self.founds])
        right_counter = Counter([x[0] for x in self.rights])
        for type_, count in origin_counter.items():
            # 每个实体类型的效果
            origin = count
            found = found_counter.get(type_, 0)
            right = right_counter.get(type_, 0)
            recall, precision, f1 = self.compute(origin, found, right)
            class_info[type_] = {"acc": round(precision, 4), 'recall': round(recall, 4), 'f1': round(f1, 4)}
        origin = len(self.origins)
        found = len(self.founds)
        right = len(self.rights)
        recall, precision, f1 = self.compute(origin, found, right)
        return {'acc': precision, 'recall': recall, 'f1': f1}, class_info

    def update(self, label_paths, pred_paths):
        '''
        :param label_paths: [[],[],[],....]
        :param pred_paths: [[],[],[],.....]
        :return
        Example:
            >>> labels_paths = [['O', 'O', 'O', 'B-MISC', 'I-MISC', 'I-MISC', 'O'], ['B-PER', 'I-PER', 'O']]
            >>> pred_paths = [['O', 'O', 'B-MISC', 'I-MISC', 'I-MISC', 'I-MISC', 'O'], ['B-PER', 'I-PER', 'O']]
        '''
        for label_path, pre_path in zip(label_paths, pred_paths):
            # label_path是原始的标签序列, pre_path是预测的标签序列
            # 因为NER的评估指标是Entity级别的, 因此需要将标签序列转换为Entity级别的标签序列
            # ['B-PER', 'I-PER', 'O', 'B-LOC']
            # [['PER', 0,1], ['LOC', 3, 3]]
            label_entities = get_entities(label_path, self.id2label)
            pre_entities = get_entities(pre_path, self.id2label)
            
            self.origins.extend(label_entities)
            self.founds.extend(pre_entities)
            self.rights.extend([pre_entity for pre_entity in pre_entities if pre_entity in label_entities])