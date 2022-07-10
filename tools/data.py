"""
处理NER数据的常用Dataset类
"""

from os import truncate
import torch
from tools.data_utils import align_label_example


def collate_fn(batch):
    """
    将list形式组织的样本数据拼接成一个batch

    :param batch (list): 是一个列表, 每个元素是dataset中的一个样本
    :return new_batch (dict): 是一个字典, 包含了一个batch的数据
    """

    all_input_ids, all_attention_mask, all_labels, all_lens = [], [], [], []
    for dataset in batch:
        # dataset中一个样本的数据
        input_ids, attention_mask = dataset['input_ids'], dataset['attention_mask']
        labels, sent_length = dataset['labels'], dataset['sent_length']
        all_input_ids.append(input_ids)
        all_attention_mask.append(attention_mask)
        all_labels.append(labels)
        all_lens.append(sent_length)
    
    # list装载tensor, 可以用stack方法将其转换为tensor
    all_input_ids = torch.stack(all_input_ids)
    all_attention_mask = torch.stack(all_attention_mask)
    all_labels = torch.stack(all_labels)
    all_lens = torch.stack(all_lens)
    
    # 该batch中最长的句子的长度
    max_len = max(all_lens).item()
    all_input_ids = all_input_ids[:, :max_len]
    all_attention_mask = all_attention_mask[:, :max_len]
    all_labels = all_labels[:,:max_len]

    new_batch = {'input_ids': all_input_ids, 'attention_mask': all_attention_mask, 'labels': all_labels, 'sent_length': all_lens}
    
    return new_batch


class SequenceLabelingDataset(torch.utils.data.Dataset):
    
    def __init__(self, data, labels_to_ids, tokenizer, max_length=128):
        """
        :param data (list): 每个样本是包含text和labels的dict
        :param labels_to_ids: 字典, key为标签, value为标签的id
        :param tokenizer: 用于处理输入句子的tokenizer
        :param max_length: 限制句子的最大长度
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
        # 句子的真实长度
        item['sent_length'] = torch.as_tensor(len(tokenized_token_list))

        return item


def pred_collate_fn(batch):
    """
    将list形式组织的样本数据拼接成一个batch

    :param batch (list): 是一个列表, 每个元素是dataset中的一个样本
    :return new_batch (dict): 是一个字典, 包含了一个batch的数据
    """

    all_input_ids, all_attention_mask, all_lens = [], [], []
    for dataset in batch:
        # dataset中一个样本的数据
        input_ids, attention_mask = dataset['input_ids'], dataset['attention_mask']
        sent_length = dataset['sent_length']
        all_input_ids.append(input_ids)
        all_attention_mask.append(attention_mask)
        all_lens.append(sent_length)
    
    # list装载tensor, 可以用stack方法将其转换为tensor
    all_input_ids = torch.stack(all_input_ids)
    all_attention_mask = torch.stack(all_attention_mask)
    all_lens = torch.stack(all_lens)
    
    # 该batch中最长的句子的长度
    max_len = max(all_lens).item()
    all_input_ids = all_input_ids[:, :max_len]
    all_attention_mask = all_attention_mask[:, :max_len]

    new_batch = {'input_ids': all_input_ids, 'attention_mask': all_attention_mask, 'sent_length': all_lens}
    
    return new_batch


class PredSequenceLabelingDataset(torch.utils.data.Dataset):
    
    def __init__(self, data, tokenizer, max_length=128):
        """
        :param data (list): 每个样本是包含text的dict
        :param tokenizer: 用于处理输入句子的tokenizer
        :param max_length: 限制句子的最大长度
        """
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):

        sample = self.data[idx]
        # sentence就是原始的输入句子, tags是对应的标签序列转换成字符串(中间用空格分开)
        sentence = sample['text']
        token_list = [token for token in sentence]
        tokenized_token_list = []

        for token in token_list:
            # tokenize the word and count the number of subwords
            tokenized_token = self.tokenizer.tokenize(token)
            tokenized_token_list.extend(tokenized_token)

        # 将token_list转换为tensor
        sent_encoded = self.tokenizer.encode_plus(tokenized_token_list,    
            add_special_tokens=True,  # Add [CLS] and [SEP]
            max_length = self.max_length,  # maximum length of a sentence
            truncation=True,
            padding='max_length',
            return_attention_mask = True,  # Generate the attention mask
        )

        # 将所有内容转换为tensor
        item = {key: torch.as_tensor(val) for key, val in sent_encoded.items()}
        # 句子的真实长度
        item['sent_length'] = torch.as_tensor(len(tokenized_token_list))

        return item