import torch.nn as nn
import torch

class Config:
    def __init__(self):
        """
        一些参数设置
        """

        self.bert_model = "bert-base-chinese"
        self.epochs = 2
        self.batch_size = 16
        self.learning_rate = 2e-5
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")