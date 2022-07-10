from tkinter.messagebox import NO
import torch.nn as nn
import torch

class Config:
    def __init__(self, args):
        """
        一些参数设置
        
        args: 命令行参数
        """

        self.pretrained_model_name = "bert-base-chinese"
        self.data_dir = "./data/cluener/conll_format"
        # 标签信息
        self.unique_tags = []
        self.labels_to_ids = {}
        self.ids_to_labels = {}
        self.epochs = 2
        self.logger = None
        self.batch_size = 16
        self.log_step = 50  # 每隔多少步(batch)打印一次日志
        self.learning_rate = 2e-5
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        for k, v in args.__dict__.items():
            if v is not None:
                self.__dict__[k] = v
    
    def __repr__(self):
        return "{}".format(self.__dict__.items())