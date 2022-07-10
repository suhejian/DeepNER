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
from metrics.metric_utils import get_entities
from metrics.ner_metrics import SequenceLabelingEntityScore
from tqdm import tqdm


class Trainer(object):
    def __init__(self, model, opt):
        """
        初始化
        """

        self.model = model
        # 模型预测结果是标签索引, 为了方便评估模型效果, 有必要把标签索引转换为标签名称
        self.ids2labels = opt.ids_to_labels
        self.opt = opt
        self.logger = opt.logger
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = transformers.AdamW(model.parameters(), lr=opt.learning_rate)
    
    def train(self, data_loader, epoch):
        self.model.train()
        loss_list = []

        self.logger.info(f"**********Start Training Epoch {epoch}**********")
        for step, batch in enumerate(data_loader):
            
            input_ids = batch['input_ids'].to(self.opt.device)  # [batch_size, seq_len]
            attention_masks = batch['attention_mask'].to(self.opt.device)   # [batch_size, seq_len]
            labels = batch['labels'].to(self.opt.device)    # [batch_size, seq_len]
            sent_lengths = batch['sent_length']  # [batch_size]
            # 训练的时候肯定是有标签的
            output = self.model(input_ids=input_ids, attention_mask=attention_masks, labels=labels)
            loss, logits = output.loss, output.logits

            # 反向传播
            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()
            if step % self.opt.log_step == 0:
                self.logger.info(f"Train Loss Per Batch: {loss.item()}")
            loss_list.append(loss.cpu().item())
        
        self.logger.info(f"Train Loss in Epoch {epoch}: {np.mean(loss_list)}")

    def eval(self, data_loader, epoch, is_test=False):
        self.model.eval()
        data_name = "Test Dataset" if is_test else "Dev Dataset"
        self.logger.info(f"***************Start Evaluating in {data_name}*******************")
        
        # 由于要评估模型效果, 所以就需要加载相应的评估类
        metric = SequenceLabelingEntityScore(id2label=self.ids2labels)
        
        eval_loss = 0.0
        for i, batch in enumerate(data_loader):
            input_ids = batch['input_ids'].to(self.opt.device)  # [batch_size, seq_len]
            attention_masks = batch['attention_mask'].to(self.opt.device)   # [batch_size, seq_len]
            labels = batch['labels'].to(self.opt.device)    # [batch_size, seq_len]
            sent_lengths = batch['sent_length']  # [batch_size]
            
            # 评估的时候不计算梯度
            with torch.no_grad():
                output = self.model(input_ids=input_ids, attention_mask=attention_masks, labels=labels)
            
            tmp_eval_loss, logits = output.loss, output.logits
            eval_loss += tmp_eval_loss.item()

            # 预测结果, logits: [batch_size, seq_len, num_classes]
            preds = np.argmax(logits.cpu().numpy(), axis=2).tolist()
            out_label_ids = labels.cpu().numpy().tolist()
            input_lens = sent_lengths.cpu().numpy().tolist()    # [batch_size], 句子的真实长度
            for i, label in enumerate(out_label_ids):
                # 对应每一个样本的标签序列
                temp_1 = [] # 存放真实标签序列
                temp_2 = [] # 存放预测标签序列
                for j, label_id in enumerate(label):
                    # 对应每一个token的标签
                    if j == 0:  # 句子开头, [CLS]
                        continue
                    elif j == input_lens[i] - 1:    # 句子结尾, 该句子的标签序列和预测序列已经得到
                        metric.update(pred_paths=[temp_2], label_paths=[temp_1])
                        break
                    else:
                        temp_1.append(self.ids2labels[label_id])
                        temp_2.append(self.ids2labels[preds[i][j]])
        # 整个数据集的评估结果
        eval_info, entity_info = metric.result()
        results = {f'{key}': value for key, value in eval_info.items()}
        results['loss'] = eval_loss
        self.logger.info(f"*****Evaluation Results in Epoch {epoch}*****")
        info = "-".join([f' {key}: {value:.4f} ' for key, value in results.items()])
        self.logger.info(info)
        self.logger.info(f"***** Entity results in Epoch {epoch} *****")
        for entity_type in sorted(entity_info.keys()):
            self.logger.info(entity_type)
            info = "-".join([f' {key}: {value:.4f} ' for key, value in entity_info[entity_type].items()])
            self.logger.info(info)
        # results (dict): loss, f1, p, r
        return results

    
    def predict(self, data_loader):
        self.model.eval()
        self.logger.info("Start Predicting...")
        
        pred_result = []
        for batch in tqdm(data_loader, desc="Predicting"):
            input_ids = batch['input_ids'].to(self.opt.device)  # [batch_size, seq_len]
            attention_masks = batch['attention_mask'].to(self.opt.device)   # [batch_size, seq_len]
            labels = batch['labels'].to(self.opt.device)    # [batch_size, seq_len]
            sent_lengths = batch['sent_length']  # [batch_size]
            
            # 预测的时候不计算梯度
            with torch.no_grad():
                output = self.model(input_ids=input_ids, attention_mask=attention_masks, labels=labels)
            
            _, logits = output.loss, output.logits
            # 预测结果, logits: [batch_size, seq_len, num_classes]
            preds = logits.detach().cpu().numpy()
            preds = np.argmax(preds, axis=2).tolist()   # [batch_size, seq_len]
            input_lens = sent_lengths.cpu().numpy().tolist()    
            for i, label in enumerate(preds):
                # 对应每一个样本的标签序列
                temp = []
                for j, label_id in enumerate(label):
                    # 对应每一个token的标签id
                    if j == 0:  # 句子开头, [CLS]
                        continue
                    elif j == input_lens[i] - 1:    # 句子结尾, 该句子的标签序列和预测序列已经得到
                        pred_result.append(get_entities(temp, self.ids2labels))
                        break
                    else:
                        temp.append(label_id)
                        
        with open(self.opt.predict_path, 'w') as f:
            for pred in pred_result:
                f.write(str(pred) + '\n')
        self.logger.info("Predicting Done.")
        self.logger.info(f"Predict result save to {self.opt.predict_path}")
        
        return pred_result

    
    def save(self, path):
        torch.save(self.model.state_dict(), path)

    
    def load(self, path):
        self.model.load_state_dict(torch.load(path))





