"""
处理NER数据的常用Dataset类
"""

from os import truncate
import torch
from tools.data_utils import align_label_example

class SequenceLabelingDataset(torch.utils.data.Dataset):
    
    def __init__(self, data, labels_to_ids, tokenizer, max_length=128):
        """
        :param data (list): 每个样本是包含text和labels的dict
        :param labels_to_ids: 字典, key为标签, value为标签的id
        :param tokenizer: 用于处理输入句子的tokenizer
        """
        self.data = data
        self.labels_to_ids = labels_to_ids
        self.tokenizer = tokenizer
        self.max_length = max_length
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):

        sample = self.data[idx]
        # sentence就是原始的输入句子, tags是对应的标签序列转换成字符串(中间用空格分开)
        sentence = sample['text']
        token_list = [token for token in sentence]
        tag_list = sample['labels']
        
        # 经过align_label_example处理后的token_list和tag_list
        tokenized_token_list, aligned_tag_list = align_label_example(token_list, tag_list, self.tokenizer)

        # 将token_list和tag_list转换为tensor
        sent_encoded = self.tokenizer.encode_plus(tokenized_token_list,    
            add_special_tokens=True,  # Add [CLS] and [SEP]
            max_length = self.max_length,  # maximum length of a sentence
            truncation=True,
            padding='max_length',
            return_attention_mask = True,  # Generate the attention mask
        )

        # 得到每个标签对应的id
        label_ids = [-100] * self.max_length
        for i, tag in enumerate(aligned_tag_list):
            if self.labels_to_ids.get(tag) != None:
                # 之所以是i + 1是因为第一个token是[CLS]
                label_ids[i + 1] = self.labels_to_ids.get(tag)

        # 将所有内容转换为tensor
        item = {key: torch.as_tensor(val) for key, val in sent_encoded.items()}
        item['labels'] = torch.as_tensor(label_ids)

        return item
