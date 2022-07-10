import argparse
import pandas as pd
from tqdm import tqdm
from transformers import BertTokenizer, BertForTokenClassification
from torch.utils.data import DataLoader
from tools.data_utils import get_unique_tags
from tools.data import SequenceLabelingDataset, collate_fn
from tools.data_utils import read_data
from torch.utils.data import RandomSampler, SequentialSampler
from train_and_eval import Trainer
import torch
from tools.utils import get_logger
from config import Config


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default='./data/cluener/conll_format')
    parser.add_argument("--log_path", type=str, default='./logs/cluener.log')
    parser.add_argument("--save_path", type=str, default="./save_models/cluener.pt")
    parser.add_argument("--predict_path", type=str, default="./predict_results/predict.txt")

    args = parser.parse_args()
    print(args)
    opt = Config(args)
    logger = get_logger(opt.log_path)
    
    # 加载数据
    train_data, dev_data = read_data(opt.data_dir)

    # 得到标签相关信息, 一方面为了将标签转换为索引, 另一方面为BERT初始化时使用
    unique_tags, labels_to_ids, ids_to_labels = get_unique_tags(train_data)
    opt.unique_tags = unique_tags
    opt.labels_to_ids = labels_to_ids
    opt.ids_to_labels = ids_to_labels

    opt.logger = logger

    # tokenizer用于处理文本
    tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')

    train_dataset = SequenceLabelingDataset(data=train_data, labels_to_ids=labels_to_ids, tokenizer=tokenizer, max_length=128)
    dev_dataset = SequenceLabelingDataset(data=dev_data, labels_to_ids=labels_to_ids, tokenizer=tokenizer, max_length=128)

    train_sampler = RandomSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, batch_size=32, sampler=train_sampler, collate_fn=collate_fn)

    dev_sampler = SequentialSampler(dev_dataset)
    dev_dataloader = DataLoader(dev_dataset, batch_size=32, sampler=dev_sampler, collate_fn=collate_fn)

    # 模型
    model = BertForTokenClassification.from_pretrained("bert-base-chinese", num_labels=len(opt.unique_tags))
    # 设置设备, 数据已经移动到config中设置的设备上
    # 因此需要将模型也移动到相同设备上
    model = model.to(opt.device)
    # 模型训练
    trainer = Trainer(model=model, opt=opt)

    best_f1 = 0
    best_test_f1 = 0
    for i in range(opt.epochs):
        opt.logger.info(f"epoch: {i}")
        trainer.train(data_loader=train_dataloader, epoch=i)
        results = trainer.eval(data_loader=dev_dataloader, epoch=i)
        # results包含loss, acc, recall, f1
        f1 = results["f1"]
        if f1 > best_f1:
            best_f1 = f1
            trainer.save(opt.save_path)

    pred_result = trainer.predict(data_loader=dev_dataloader)