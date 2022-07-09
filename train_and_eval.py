from cProfile import label
from statistics import mode
import numpy as np
import prettytable as pt
import torch
import torch.autograd
import torch.nn as nn
import transformers
from sklearn.metrics import precision_recall_fscore_support, f1_score
from torch.utils.data import DataLoader
from tqdm import tqdm


class Trainer(object):
    def __init__(self, model, opt):
        """
        初始化
        """

        self.model = model
        self.opt = opt
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = transformers.AdamW(model.parameters(), lr=opt.learning_rate)
    
    def train(self, data_loader, epoch):
        self.model.train()
        loss_list = []
        pred_result = []
        label_result = []

        for batch in tqdm(data_loader):

            input_ids = batch['input_ids'].to(self.opt.device)  # [batch_size, seq_len]
            attention_masks = batch['attention_mask'].to(self.opt.device)   # [batch_size, seq_len]
            labels = batch['labels'].to(self.opt.device)    # [batch_size, seq_len]

            output = self.model(input_ids=input_ids, attention_mask=attention_masks, labels=labels)
            loss, logits = output.loss, output.logits
            num_classes = logits.shape[-1]  # tag数目
            # logits: [batch_size, seq_len, num_classes]

            # padding的部分, 不计入损失计算
            active_loss = attention_masks.view(-1) == 1
            active_logits = logits.view(-1, num_classes)[active_loss]
            active_pred = torch.argmax(active_logits, dim=-1)
            # print(active_logits.shape)
            active_labels = labels.view(-1)[active_loss]
            # print(active_labels.shape)
            true_loss = self.criterion(active_logits, active_labels)

            # 反向传播
            true_loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()

            print(f"Train Loss: {true_loss.item()}")
            loss_list.append(true_loss.cpu().item())

            label_result.append(active_labels)
            pred_result.append(active_pred)
        
        # label_result = torch.cat(label_result)
        # pred_result = torch.cat(pred_result)

        # 计算precision, recall和f1
        # p, r, f1, _ = precision_recall_fscore_support(label_result.cpu().numpy(),
        #                                                 pred_result.cpu().numpy(),
        #                                                 average="macro")
        
        # # 这个只是Label层级的计算指标, NER实际使用的指标是Entity级别的
        # table = pt.PrettyTable(["Train {}".format(epoch), "Loss", "F1", "Precision", "Recall"])
        # table.add_row(["Label", "{:.4f}".format(np.mean(loss_list))] + 
        #                 ["{:3.4f}".format(x) for x in [f1, p, r]])
        # print(table)
        # return f1

    # def eval(self, data_loader, model, epoch, is_test=False):
    #     self.model.eval()

    #     with torch.no_grad():




