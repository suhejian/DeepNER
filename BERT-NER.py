# %%
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)


import torch
from torch import nn
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# %%
from transformers import BertModel
bert = BertModel.from_pretrained('bert-base-uncased')

# %%
from transformers import BertTokenizer,BertForTokenClassification
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
tokenizer.add_tokens(['B_geo','I_geo','B_per','I_per','B_org','I_org'])

# %%
data = pd.read_csv('./data/ner.csv')

# %%
data.head()

# %%
def preprocess_dataset(data): 
    '''
        Here we will remove all the tags except for 'Org','Geo', and 'Per'
        type. These three are our targeted Entities.
    '''
#     data['labels'] = data['labels'].str.replace('-','_')
    data['labels'] = data['labels'].str.replace('B_gpe','O')
    data['labels'] = data['labels'].str.replace('I_gpe','O')
    data['labels'] = data['labels'].str.replace('B_tim','O')
    data['labels'] = data['labels'].str.replace('I_tim','O')
    data['labels'] = data['labels'].str.replace('B_eve','O')
    data['labels'] = data['labels'].str.replace('I_eve','O')
    data['labels'] = data['labels'].str.replace('B_nat','O')
    data['labels'] = data['labels'].str.replace('I_nat','O')
    data['labels'] = data['labels'].str.replace('B_art','O')
    data['labels'] = data['labels'].str.replace('I_art','O')
    
    return data

# %%
data=preprocess_dataset(data)

# %%
'''
This is to remove the sentences that doesn't contain our targeted entities.
'''
sum=0
for index, i in enumerate(data['labels']):
    a=set(i.split(' '))
    if(len(a)<=1):
        data.drop(labels=index, axis=0,inplace=True)
        sum+=1    
print(sum)
data = data.dropna()
data.reset_index(drop=True, inplace=True)

# %%
unique_tags = data.labels.apply(lambda x: pd.value_counts(x.split(" "))).sum(axis = 0)
unique_tags

# %%
label_to_ids = {'B-geo': 1,
 'B-org': 2,
 'B-per': 3,
 'I-geo': 4,
 'I-org': 5,
 'I-per': 6,
 'O': 0}

# %%
ids_to_label = {1:'B-geo',
 2:'B-org',
 3:'B-per',
 4:'I-geo',
 5:'I-org',
 6:'I-per',
 0:'O'}

# %%
def tokenize_and_preserve_labels(sentence, text_labels, tokenizer):

    """
    Word piece tokenization makes it difficult to match word labels
    back up with individual word pieces. This function tokenizes each
    word one at a time so that it is easier to preserve the correct
    label for each subword. It is, of course, a bit slower in processing
    time, but it will help our model achieve higher accuracy.
    """

    tokenized_sentence = []
    labels = []
    
#     print('##')
#     print(sentence)
#     print(text_labels)
#     print('###')
    for word, label in zip(sentence, text_labels):

        # Tokenize the word and count # of subwords the word is broken into
        tokenized_word = tokenizer.tokenize(word)
        n_subwords = len(tokenized_word)
        
        ## if sentence consist of more than 125 words, discard the later words.
        if(len(tokenized_sentence)>=125):
            return tokenized_sentence, labels
        # Add the tokenized word to the final tokenized word list
        tokenized_sentence.extend(tokenized_word)

        # Add the same label to the new list of labels `n_subwords` times
        labels.extend([label] * n_subwords)

    return tokenized_sentence, labels

# %%
class Ner_Data(Dataset):

    def __init__(self, data):
        self.data = data
#         print("dataloader initialized")
        
        

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
#         print(idx)
        sentence = self.data['text'][idx].strip().split()  
        word_labels = self.data['labels'][idx].split(" ") 
#         print(len(sentence))
#         if(len(sentence)>64):
#             sentence=sentence[:63]
#             word_labels=word_labels[:63]
# #         print(sentence)
        t_sen, t_labl = tokenize_and_preserve_labels(sentence, word_labels, tokenizer)
                
        sen_code = tokenizer.encode_plus(t_sen,    
            add_special_tokens=True,  # Add [CLS] and [SEP]
            max_length = 128,  # maximum length of a sentence
            pad_to_max_length=True,  # Add [PAD]s
            return_attention_mask = True,  # Generate the attention mask
#             return_tensors = 'pt'
            )
             
            
        labels = [-100]*128
        for i, tok in enumerate(t_labl):
#             tok = tokenizer.convert_ids_to_tokens(i)
#             print(tok)
#             print(tok)
#             print(label_to_ids.get(tok))
            if label_to_ids.get(tok) != None:
                labels[i+1]=label_to_ids.get(tok)

        # step 4: turn everything into PyTorch tensors
        item = {key: torch.as_tensor(val) for key, val in sen_code.items()}
        item['labels'] = torch.as_tensor(labels)

        return item


# %%
train_data = Ner_Data(data)

# %%
print(len(train_data[10]['input_ids']))
print(len(train_data[10]['labels']))
print(train_data[10]['labels'])

# %%
train_data[148]
print(train_data[148]['input_ids'].detach().numpy())
print(tokenizer.convert_ids_to_tokens(train_data[148]['input_ids'].detach().numpy()))

print('#####')
for i in train_data[148]['labels'].detach().numpy():
#     print(i)
    print(ids_to_label.get(i))

# %%
train_dataloader = DataLoader(train_data, batch_size=32, shuffle=False)

# %%
model2 = model =BertForTokenClassification.from_pretrained('bert-base-uncased', num_labels=len(label_to_ids))
model2.to(device)

# %%
learning_rate = 0.0001
batch_size = 64
epochs = 5

# %%
loss_fn2 = nn.CrossEntropyLoss()

# %%
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# %%
def train_loop(train_dataloader, model, optimizer):
    size = len(train_dataloader.dataset)
    train_loss =0
    for i,sample in enumerate(train_dataloader):
        optimizer.zero_grad()
#         print(sample)
        ids=sample['input_ids'].to(device)
        mask=sample['attention_mask'].to(device)
        labels = sample['labels'].to(device)
        pred = model2(input_ids=ids, attention_mask=mask ,labels = labels )
        loss = pred[0]
        
#         print(f"loss: {loss.item()}")
        train_loss+=loss.item()
        # Backpropagation
        loss.backward()
        optimizer.step()
        
        
        if(i>0 and i % 500==0):
            print(f"loss: {train_loss/i:>4f}  [{i:>5d}/{size/32}]")
    return train_loss

# %%
epochs = 6
train_loss = []
for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    loss = train_loop(train_dataloader, model, optimizer)
    train_loss.append(loss)
#     test_loop(test_dataloader, model, loss_fn)
print("Done!")

# %%
test_sen = data[200:220]
test_sen = test_sen.reset_index(drop=True)
test_sen

# %%
class process_sentence_single(Dataset):

    def __init__(self, text):
        self.text = text
        print("dataloader initialized")
        
    def __len__(self):
        return 1

    def __getitem__(self,idx):

        sentence = self.text.strip().split() 
        
        tokenized_sentence = []

        for word in sentence:
            # Tokenize the word and count # of subwords the word is broken into
            tokenized_word = tokenizer.tokenize(word)
            tokenized_sentence.extend(tokenized_word)
        
        
        sen_code = tokenizer.encode_plus(tokenized_sentence,    
            add_special_tokens=True,  # Add [CLS] and [SEP]
#             max_length = 128,  # maximum length of a sentence
#             pad_to_max_length=True,  # Add [PAD]s
            return_attention_mask = True,  # Generate the attention mask
#             return_tensors = 'pt'
            )

        # step 4: turn everything into PyTorch tensors
        item = {key: torch.as_tensor(val) for key, val in sen_code.items()}
        return item

# %%
class process_sentence_batch(Dataset):

    def __init__(self, data):
        self.data = data
#         print("dataloader initialized")
        
    def __len__(self):
        return len(self.data)

    def __getitem__(self,idx):

        sentence = self.data['text'][idx].strip().split() 
        
        tokenized_sentence = []

        for word in sentence:
            # Tokenize the word and count # of subwords the word is broken into
            tokenized_word = tokenizer.tokenize(word)
            tokenized_sentence.extend(tokenized_word)
        
        
        sen_code = tokenizer.encode_plus(tokenized_sentence,    
            add_special_tokens=True,  # Add [CLS] and [SEP]
            max_length = 128,  # maximum length of a sentence
            pad_to_max_length=True,  # Add [PAD]s
            return_attention_mask = True,  # Generate the attention mask
#             return_tensors = 'pt'
            )

        # step 4: turn everything into PyTorch tensors
        item = {key: torch.as_tensor(val) for key, val in sen_code.items()}
        return item

# %%
def infer_text(test_datas_batch, model):
    for i,sample in enumerate(test_datas_batch):
        ids=sample['input_ids'].to(device)
        mask=sample['attention_mask'].to(device)
        pred = model2(input_ids=ids, attention_mask=mask)

        return ids, pred

# %%
all_things=[]
def make_batch_pred(test_sen):
    pre_text = process_sentence_batch(test_sen)
    test_datas_batch = DataLoader(pre_text, batch_size = 8, shuffle=False)

    ids, preds = infer_text(test_datas_batch, model2)


    flattened_predictions = []
    for logit in preds['logits']:
        active_logits = logit.view(-1, model2.num_labels) # shape (batch_size * seq_len, num_labels)
        flattened_predictions.append(torch.argmax(active_logits, axis=1)) # shape (batch_size * seq_len,)
    # flattened_predictions

    for i, predict in  enumerate(flattened_predictions):
        text_tokens= tokenizer.convert_ids_to_tokens(ids[i])
        sep_i = text_tokens.index('[SEP]')
        text_labels = []
        for i in predict.squeeze(0).cpu().numpy():
            text_labels.append(ids_to_label.get(i))


        text_tokens = text_tokens[1:sep_i]
        text_labels = text_labels[1:sep_i]


        print("\n printing tokens with labels")
        print(text_tokens)
        print(text_labels)
        
        print(len(text_tokens))
        print(len(text_labels))

        sent = []
        
        for text in text_tokens:
            if text.startswith('##'):
                sent[-1] = sent[-1]+text[2:]
            
        
        per=[]
        geo=[]
        org=[]

        for text, label in zip(text_tokens,text_labels):
            print(text,label)

            if(label[2:] == 'per'):
                if text.startswith('##'):
                    per[-1] = per[-1]+text[2:]
                else:
                    per.append(text)

            if(label[2:] == 'geo'):
                if text.startswith('##'):
                    geo[-1] = geo[-1]+text[2:]
                else:
                    geo.append(text)

            if(label[2:] == 'org'):
                if text.startswith('##'):
                    org[-1] = org[-1]+text[2:]
                else:
                    org.append(text)
        
        all_things.append({'sent':sent,
            'per':per, 
            'geo':geo,
            'org':org})

# %%

def make_single_pred(sentence):

    # get the processed input_ids and mask
    # test_text = "Mark is the ceo of Facebook. located in California ."
    test_text = sentence
    pre_text = process_sentence_single(test_text)
    text= pre_text[0]

    ids = text ['input_ids']
    mask = text ['attention_mask']

    
    #make prediction
    
    test_pred = model2(input_ids=torch.unsqueeze(ids,0).to(device), attention_mask=torch.unsqueeze(mask,0).to(device))

    
    ## flatten prediction
    active_logits = test_pred[0].view(-1, model.num_labels) # shape (batch_size * seq_len, num_labels)
    flattened_predictions = torch.argmax(active_logits, axis=1) # shape (batch_size * seq_len,)
    print("\nFlatten Predictions.....\n")
    print(flattened_predictions)

    
    
    print("\n printing tokens.....")
    for i in torch.unsqueeze(ids,0):
        print(tokenizer.convert_ids_to_tokens(i))

    # convert ids to corresponding tokens
    text_tokens= tokenizer.convert_ids_to_tokens(ids)

    # convert predctions to labels
    text_labels = []
    for i in flattened_predictions.squeeze(0).cpu().numpy():
        text_labels.append(ids_to_label.get(i))

#     print("\n printing predicted token labels.....")
#     print(text_labels)

    # remove first and last tokens ([CLS] and [SEP])
    text_tokens = text_tokens[1:-1]
    text_labels = text_labels[1:-1]


    print("\n printing tokens with labels")
    print(text_tokens)
    print(text_labels)
    
    return text_tokens, text_labels
#     print("\n printing zipped tokens with labels.....\n")
#     for token, label in zip(text_tokens,text_labels):
#         print(token,label)

# %%
txt, lbl = make_single_pred("Sundar Pichai lived in India is CEO of Google .")

# %%
df = pd.DataFrame(columns = ['Free flow of Text','Extracted Name','Extracted Location','Extracted Organization'])
df

# %%
for sent in test_sen['text']:
#     print(sent)
    text_sen = sent
    per=[]
    geo=[]
    org=[]
    txt, lbl = make_single_pred(sent)
    for text, label in zip(txt,lbl):
#         print(text,label)
        
        if(label == 'I-per'):
            if not text.startswith('##'):
#                 print("####")
#                 print(text)
                if(len(per)<=0):
                    per.append(text)
                else:
                    per[-1] = per[-1]+' '+ text
                continue

        if(label[2:] == 'per'):
            if text.startswith('##'):
                per[-1] = per[-1]+text[2:]
            else:
                per.append(text)

                
                
        if(label == 'I-geo'):
            
            if not text.startswith('##'):
#                 print("####")
#                 print(text)
                if(len(geo)<=0):
                    geo.append(text)
                else:
                    geo[-1] = geo[-1]+' '+ text
                
                continue
                
        if(label[2:] == 'geo'):
            if text.startswith('##'):
                geo[-1] = geo[-1]+text[2:]
            else:
                geo.append(text)

                
                
        if(label == 'I-org'):
            if not text.startswith('##'):
#                 print("####")
#                 print(text)
                if(len(org)<=0):
                    org.append(text)
                else:
                    org[-1] = org[-1]+' '+ text
                continue
                
        if(label[2:] == 'org'):
            if text.startswith('##'):
                org[-1] = org[-1]+text[2:]
            else:
                org.append(text)
                
#     df.append({'Free flow of Text':text_sen, 'Extracted Name':per, 'Extracted Location':geo,'Extracted Organization':org}, ignore_index=True)
        
    new_record = pd.DataFrame([[text_sen,per,geo,org]], columns = ['Free flow of Text','Extracted Name','Extracted Location','Extracted Organization'])

    df = pd.concat([df, new_record])

# %%
df.reset_index(drop=True, inplace=True)
df

# %%
from seqeval.metrics import f1_score, accuracy_score, precision_score, recall_score

# %%
train_size = 0.8
train_dataset = data.sample(frac=train_size,random_state=200)
test_dataset = data.drop(train_dataset.index).reset_index(drop=True)
train_dataset = train_dataset.reset_index(drop=True)

# %%
test_data = Ner_Data(test_dataset)
test_data_loader = DataLoader(test_data, batch_size=32, shuffle=False)

# %%
def valid(model, testing_loader):
    model.eval()
    eval_loss, eval_accuracy = 0, 0
    nb_eval_examples, nb_eval_steps = 0, 0
    eval_preds, eval_labels = [], []
    
    with torch.no_grad():
        for idx, batch in enumerate(testing_loader):
            
            ids = batch['input_ids'].to(device, dtype = torch.long)
            mask = batch['attention_mask'].to(device, dtype = torch.long)
            labels = batch['labels'].to(device, dtype = torch.long)
            preds= model2(input_ids=ids, attention_mask=mask, labels=labels)

            loss = preds['loss']
            eval_logits = preds['logits'] 
            
            eval_loss += loss.item()

            nb_eval_steps += 1
            nb_eval_examples += labels.size(0)
        
            if idx % 100==0:
                loss_step = eval_loss/nb_eval_steps
                print(f"Validation loss per 100 evaluation steps: {loss_step}")
              
            # compute evaluation accuracy
            flattened_targets = labels.view(-1) # shape (batch_size * seq_len,)
            active_logits = eval_logits.view(-1, model.num_labels) # shape (batch_size * seq_len, num_labels)
            flattened_predictions = torch.argmax(active_logits, axis=1) # shape (batch_size * seq_len,)
            
            # only compute accuracy at active labels
            active_accuracy = labels.view(-1) != -100 # shape (batch_size, seq_len)
        
            labels = torch.masked_select(flattened_targets, active_accuracy)
            predictions = torch.masked_select(flattened_predictions, active_accuracy)
            
            eval_labels.extend(labels)
            eval_preds.extend(predictions)
            
            tmp_eval_accuracy = accuracy_score(labels.cpu().numpy(), predictions.cpu().numpy())
            eval_accuracy += tmp_eval_accuracy

    labels = [ids_to_label[id.item()] for id in eval_labels]
    predictions = [ids_to_label[id.item()] for id in eval_preds]
    
    eval_loss = eval_loss / nb_eval_steps
    eval_accuracy = eval_accuracy / nb_eval_steps
    print(f"Validation Loss: {eval_loss}")
    print(f"Validation Accuracy: {eval_accuracy}")

    return labels, predictions

# %%
labels, predictions = valid(model2, test_data_loader)

# %%
from seqeval.metrics import classification_report

print(classification_report([labels], [predictions]))


