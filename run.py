from statistics import mode
import pandas as pd
from tqdm import tqdm
from transformers import BertTokenizer, BertForTokenClassification
from torch.utils.data import DataLoader
from tools.data_utils import get_unique_tags
from tools.data import SequenceLabelingDataset
from tools.data_utils import read_jsonl_file
from train_and_eval import Trainer
import torch
from config import Config

# 加载数据
train_data = read_jsonl_file('./data/cluener/conll_format/train.jsonl')
dev_data = read_jsonl_file('./data/cluener/conll_format/dev.jsonl')

# 得到标签相关信息, 一方面为了将标签转换为索引, 另一方面为BERT初始化时使用
unique_tags, labels_to_ids, ids_to_labels = get_unique_tags(train_data)

# tokenizer用于处理文本
tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')

train_dataset = SequenceLabelingDataset(data=train_data, labels_to_ids=labels_to_ids, tokenizer=tokenizer, max_length=128)
dev_dataset = SequenceLabelingDataset(data=dev_data, labels_to_ids=labels_to_ids, tokenizer=tokenizer, max_length=128)

train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)
dev_dataloader = DataLoader(dev_dataset, batch_size=32, shuffle=False)

# 模型
model = BertForTokenClassification.from_pretrained("bert-base-chinese", num_labels=len(unique_tags))

# 超参数等设置
opt = Config()
# 设置设备, 数据已经移动到config中设置的设备上
# 因此需要将模型也移动到相同设备上
model = model.to(opt.device)
trainer = Trainer(model=model, opt=opt)

for i in range(opt.epochs):
    print(f"epoch: {i}")
    f1 = trainer.train(data_loader=train_dataloader, epoch=i)
    print(f1)

