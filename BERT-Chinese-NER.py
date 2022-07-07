# %% [markdown]
# 读取数据

# %%
import os

data_dir = "./data/ner_format_data/"
train_path = os.path.join(data_dir, "train.jsonl")
dev_path = os.path.join(data_dir, "dev.jsonl")
tag_path = os.path.join(data_dir, "tags.txt")

# %%
def read_data(file_path):
    """ 
    将数据读取成列表形式
    
    :param file_path: 文件路径
    :return: 数据列表
    """
    import json
    
    with open(file_path, "r", encoding="utf-8") as f:
        lines = f.readlines()
    data = [json.loads(line) for line in lines]
    
    return data

# %%
train_data = read_data(train_path)
dev_data = read_data(dev_path)

# %% [markdown]
# 建立标签与索引的映射

# %%
# 因为文本的索引可以用BERT的tokenizer来处理，而标签需要手动建立索引
with open("./data/ner_format_data/tags.txt", "r", encoding="utf-8") as f:
    tag_lines = f.readlines()
labels = [line.strip() for line in tag_lines]

unique_labels = set(labels)
# 建立标签索引
labels_to_ids = {k: v for v, k in enumerate(labels)}
ids_to_labels = {v: k for k, v in enumerate(labels)}

# %% [markdown]
# 数据预处理

# %%
# 1.tokenization
# 2.adjust the label to match the tokenization
from transformers import BertTokenizerFast

tokenizer = BertTokenizerFast.from_pretrained("bert-base-chinese")

# %% [markdown]
# 以单个样本为例

# %%
sample = train_data[0]
text = sample["text"]
label = sample["label"]
text_tokenized = tokenizer(text, padding='max_length', max_length=30, truncation=True, return_tensors="pt")
# text_tokenized # 包含input_ids, attention_mask, token_type_ids
# print(tokenizer.decode(text_tokenized.input_ids[0]))    # 输出[CLS]原文本[SEP][PAD][PAD]
# 经过tokenization后，原文本发生了变化(一方面是[SEP]的添加, 另一方面是subword的分割)
# 因此需要将标签也跟着变化
# text_tokenized.word_ids()方法可以返回每个token在原句子中的索引
word_ids = text_tokenized.word_ids()
print(tokenizer.convert_ids_to_tokens(text_tokenized['input_ids'][0]))
print(word_ids)

# %% [markdown]
# 标签对齐

# %%
def align_label_example(tokenized_input, label, labels_to_ids, label_all_tokens=True):
    """
    将标签与tokenized_input对齐
    
    :param tokenized_input: BertTokenizer处理以后的文本
    :param label: 该句子对应的标签
    :param labels_to_ids: 标签索引
    :param label_all_tokens: 是否将所有的sub word都赋予标签
    :return label_ids: 对齐后的标签
    """ 
    
    word_ids = tokenized_input.word_ids()
    
    previous_word_idx = None
    # 对齐以后的标签索引
    label_ids = []
    
    for word_idx in word_ids:
        if word_idx is None:
            label_ids.append(-100)
        elif word_idx != previous_word_idx:
            # 该subword 和 之前的subword，不属于同一个word
            try:
                label_ids.append(labels_to_ids[label[word_idx]])
            except:
                label_ids.append(-100)
        else:
            # 该subword 和 之前的subword，属于同一个word
            label_ids.append(labels_to_ids[label[word_idx]] if label_all_tokens else -100)
        previous_word_idx = word_idx
    
    return label_ids

# %%
print(f"tokenized_input: {text_tokenized}")
print(f"raw label: {label}")
print(f"aligned label: {align_label_example(text_tokenized, label, labels_to_ids)}")

# %% [markdown]
# 构造数据类

# %%
import torch

class DataSequence(torch.utils.data.Dataset):
    
    def __init__(self, data, labels_to_ids):
        
        raw_texts = [sample["text"] for sample in data]
        raw_labels = [sample["label"] for sample in data]
        self.texts = [tokenizer(text, padding='max_length', max_length=32, truncation=True, return_tensors="pt") for text in raw_texts]
        self.labels = [align_label_example(text, label, labels_to_ids) for text, label in zip(self.texts, raw_labels)]
        
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        return self.texts[idx], self.labels[idx]

# %% [markdown]
# 构造模型

# %%
from transformers import BertForTokenClassification

class BertModel(torch.nn.Module):

    def __init__(self, unique_labels):

        super(BertModel, self).__init__()

        self.bert = BertForTokenClassification.from_pretrained('bert-base-cased', num_labels=len(unique_labels))

    def forward(self, input_id, mask, label):

        output = self.bert(input_ids=input_id, attention_mask=mask, labels=label, return_dict=False)

        return output

# %% [markdown]
# 训练函数

# %%
from torch.utils.data import DataLoader
from tqdm import tqdm
from torch.optim import SGD

def train_loop(model, train_data, dev_data, labels_to_ids, batch_size=4, epochs=10):

    train_dataset = DataSequence(train_data, labels_to_ids)
    val_dataset = DataSequence(dev_data, labels_to_ids)

    train_dataloader = DataLoader(train_dataset, num_workers=4, batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, num_workers=4, batch_size=batch_size)

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    optimizer = SGD(model.parameters(), lr=LEARNING_RATE)

    if use_cuda:
        model = model.cuda()

    best_acc = 0
    best_loss = 1000

    for epoch_num in range(epochs):

        total_acc_train = 0
        total_loss_train = 0

        model.train()

        for train_data, train_label in tqdm(train_dataloader):

            train_label = train_label[0].to(device)
            mask = train_data['attention_mask'][0].to(device)
            input_id = train_data['input_ids'][0].to(device)

            optimizer.zero_grad()
            loss, logits = model(input_id, mask, train_label)

            logits_clean = logits[0][train_label != -100]
            label_clean = train_label[train_label != -100]

            predictions = logits_clean.argmax(dim=1)

            acc = (predictions == label_clean).float().mean()
            total_acc_train += acc
            total_loss_train += loss.item()

            loss.backward()
            optimizer.step()

        model.eval()

        total_acc_val = 0
        total_loss_val = 0

        for val_data, val_label in val_dataloader:

            val_label = val_label[0].to(device)
            mask = val_data['attention_mask'][0].to(device)

            input_id = val_data['input_ids'][0].to(device)

            loss, logits = model(input_id, mask, val_label)

            logits_clean = logits[0][val_label != -100]
            label_clean = val_label[val_label != -100]

            predictions = logits_clean.argmax(dim=1)          

            acc = (predictions == label_clean).float().mean()
            total_acc_val += acc
            total_loss_val += loss.item()

        val_accuracy = total_acc_val / len(dev_data)
        val_loss = total_loss_val / len(dev_data)

        print(
            f'Epochs: {epoch_num + 1} | Loss: {total_loss_train / len(train_data): .3f} | Accuracy: {total_acc_train / len(train_data): .3f} | Val_Loss: {val_loss: .3f} | Accuracy: {val_accuracy: .3f}')
        

LEARNING_RATE = 1e-2
EPOCHS = 5

model = BertModel(unique_labels=unique_labels)
train_loop(model, train_data, dev_data, labels_to_ids, batch_size=4, epochs=EPOCHS)


