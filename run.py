import argparse
import pandas as pd
import prettytable as pt
from tqdm import tqdm
from transformers import BertTokenizer, BertForTokenClassification
from torch.utils.data import DataLoader
from tools.data_utils import get_unique_tags
from tools.data import SequenceLabelingDataset, collate_fn, PredSequenceLabelingDataset, pred_collate_fn
from tools.data_utils import read_data
from torch.utils.data import RandomSampler, SequentialSampler
from train_and_eval import Trainer
import torch
from tools.data_analyze import data_length_info, data_entity_info
from tools.utils import get_logger
from config import Config


def load_data(opt):
    """
    根据设置的参数加载数据
    包括得到标签信息(如映射等)和dataset以及dataloader

    """

    # 加载数据
    train_data, dev_data, test_data = read_data(opt.data_dir)

    # 数据集信息
    train_num_sen, train_min_len, train_max_len, train_avg_len = data_length_info(train_data)
    dev_num_sen, dev_min_len, dev_max_len, dev_avg_len = data_length_info(dev_data)
    test_num_sen, test_min_len, test_max_len, test_avg_len = data_length_info(test_data)
    train_num_entities, train_entity_counter = data_entity_info(train_data)
    dev_num_entities, dev_entity_counter = data_entity_info(dev_data)
    
    table = pt.PrettyTable(["Dataset Info", 'num_sentences', "min_length", "max_length", "avg_length", "num_entities"])
    table.add_row(["Train", train_num_sen, train_min_len, train_max_len, train_avg_len, train_num_entities])
    table.add_row(["Dev", dev_num_sen, dev_min_len, dev_max_len, dev_avg_len, dev_num_entities])
    table.add_row(["Test", test_num_sen, test_min_len, test_max_len, test_avg_len, None])
    # 打印数据集的相关信息, 包括句子长度信息和实体信息
    opt.logger.info("\n{}".format(table))

    # 只有训练集和验证集有实体, 可以打印详细的实体分布情况信息
    opt.logger.info(f"Train Data Entities: {train_entity_counter}")
    opt.logger.info(f"Dev Data Entities: {dev_entity_counter}")

    # 得到标签相关信息, 一方面为了将标签转换为索引, 另一方面为BERT初始化时使用
    unique_tags, labels_to_ids, ids_to_labels = get_unique_tags(train_data)
    opt.unique_tags = unique_tags
    opt.labels_to_ids = labels_to_ids
    opt.ids_to_labels = ids_to_labels

    train_dataset = SequenceLabelingDataset(data=train_data, labels_to_ids=labels_to_ids, tokenizer=opt.tokenizer, max_length=opt.max_length)
    dev_dataset = SequenceLabelingDataset(data=dev_data, labels_to_ids=labels_to_ids, tokenizer=opt.tokenizer, max_length=opt.max_length)
    test_dataset = PredSequenceLabelingDataset(data=test_data, tokenizer=opt.tokenizer, max_length=opt.max_length)

    train_sampler = RandomSampler(train_dataset)    # 训练集抽样方式是Random的
    train_dataloader = DataLoader(train_dataset, batch_size=32, sampler=train_sampler, collate_fn=collate_fn)

    dev_sampler = SequentialSampler(dev_dataset)    # 验证集抽样方式是Sequential的
    dev_dataloader = DataLoader(dev_dataset, batch_size=32, sampler=dev_sampler, collate_fn=collate_fn)

    test_sampler = SequentialSampler(test_dataset)  # 测试集抽样方式也是Sequential的
    test_dataloader = DataLoader(test_dataset, batch_size=32, sampler=test_sampler, collate_fn=pred_collate_fn)

    return train_dataloader, dev_dataloader, test_dataloader




if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="cluener", help="数据集名称")
    parser.add_argument("--data_dir", type=str, default='./data/cluener/conll_format', help="数据集目录")
    parser.add_argument("--log_name", type=str, default='cluener.log', help="日志文件名")
    parser.add_argument("--save_name", type=str, default="bert-cluener.pt")
    parser.add_argument("--predict_name", type=str, default="cluener-predict.txt")
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch_size", type=int, default=16)


    args = parser.parse_args()
    opt = Config(args)
    opt.logger = get_logger(opt.log_path)

    # tokenizer用于处理文本
    tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
    opt.tokenizer = tokenizer

    # 加载数据
    opt.logger.info("**************************Start Loading Data*****************************")
    train_dataloader, dev_dataloader, test_dataloader = load_data(opt)
    opt.logger.info("**************************Finish Loading Dara****************************")
    opt_info = dict(opt.__dict__)
    opt.logger.info("Hyper parameters: ")
    for k in opt_info:
        opt.logger.info(f'{k}: {opt_info[k]}')

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
    # 模型对测试集进行预测
    pred_result = trainer.predict(data_loader=test_dataloader)

    # 模型可以用训练好的模型对单个句子进行预测
    trainer.load(opt.save_path)
    text = '阿森纳将在主场对阵基伏迪纳摩，赢下这场比赛他们就将铁定出线。所以，阿森纳除了缺少法布雷加斯以外，'
    print(trainer.predict_single_text(text))