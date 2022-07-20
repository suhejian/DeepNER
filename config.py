import os
import torch.nn as nn
import torch

class ConfigTest:
    def __init__(self):
        """
        用于本地测试的Config类, 不需要用到命令行参数
        """

        self.dataset = "cluener"
        self.tokenizer_name = "bert-base-chinese"
        self.pretrained_model_name = "bert-base-chinese"
        self.data_dir = "./data/cluener/conll_format"

        # 日志设置
        self.log_dir = "./logs" # 日志目录
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)
        self.log_path = os.path.join(self.log_dir, "example.log") # 日志文件路径

        # 模型结果保存设置
        self.save_dir = "./save_models/"    # 模型保存目录
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)
        self.save_path = os.path.join(self.save_dir, "model.pt") # 模型保存路径

        # 模型预测结果保存设置
        self.predict_dir = "./predict_results"    # 模型预测结果保存目录
        if not os.path.exists(self.predict_dir):
            os.makedirs(self.predict_dir)
        self.predict_path = os.path.join(self.predict_dir, "predict_result.txt") # 模型预测结果保存路径
        
        # 标签信息, 可以从外部获取
        self.unique_tags = []
        self.labels_to_ids = {}
        self.ids_to_labels = {}
        
        # 模型训练设置
        self.epochs = 2
        self.tokenizer = None
        self.max_length = 128
        self.logger = None
        self.batch_size = 16
        self.log_step = 50  # 每隔多少步(batch)打印一次日志
        self.learning_rate = 2e-5
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Config:
    def __init__(self, args):
        """
        一些参数设置
        
        args: 命令行参数
        """

        self.pretrained_model_name = "bert-base-chinese"
        self.data_dir = "./data/cluener/conll_format"

        # 日志设置
        self.log_dir = "./logs" # 日志目录
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)
        self.log_path = os.path.join(self.log_dir, args.log_name)

        # 模型结果保存设置
        self.save_dir = "./save_models/"    # 模型保存目录
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)
        self.save_path = os.path.join(self.save_dir, args.save_name)

        # 模型预测结果保存设置
        self.predict_dir = "./predict_results"    # 模型预测结果保存目录
        if not os.path.exists(self.predict_dir):
            os.makedirs(self.predict_dir)
        self.predict_path = os.path.join(self.predict_dir, args.predict_name)
        
        # 标签信息, 可以从外部获取
        self.unique_tags = []
        self.labels_to_ids = {}
        self.ids_to_labels = {}
        
        # 模型训练设置
        self.epochs = 2
        self.tokenizer = None
        self.max_length = 128
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