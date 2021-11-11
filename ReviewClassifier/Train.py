import pandas as pd
import torch
import transformers
from torch.utils.data import Dataset, DataLoader
from transformers import DistilBertModel, DistilBertTokenizer

from DistillBERTClass import *
from variables import *
from MyDataset import *
from Readcsv import df
from torch import cuda


# Setting up the device for GPU usage
device = 'cuda' if cuda.is_available() else 'cpu'

tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')


# Creating the dataset and dataloader for the neural network
train_size = 0.8
train_dataset=df.sample(frac=train_size,random_state=200)
test_dataset=df.drop(train_dataset.index).reset_index(drop=True)
train_dataset = train_dataset.reset_index(drop=True)


print("FULL Dataset: {}".format(df.shape))
print("TRAIN Dataset: {}".format(train_dataset.shape))
print("TEST Dataset: {}".format(test_dataset.shape))

training_set = MyDataset(train_dataset)
testing_set = MyDataset(test_dataset)

train_params = {'batch_size': TRAIN_BATCH_SIZE,
                'shuffle': True,
                'num_workers': 0
                }

test_params = {'batch_size': VALID_BATCH_SIZE,
                'shuffle': True,
                'num_workers': 0
                }

training_loader = DataLoader(training_set, **train_params)
testing_loader = DataLoader(testing_set, **test_params)



model = DistillBERTClass()
model.to(device)



# Function to calcuate the accuracy of the model
def calcuate_accu(big_idx, targets):
    n_correct = (big_idx==targets).sum().item()
    return n_correct

# Defining the training function on the 80% of the dataset for tuning the distilbert model

def train(epoch):
    tr_loss = 0
    n_correct = 0
    nb_tr_steps = 0
    nb_tr_examples = 0
    model.train()
    for _,data in enumerate(training_loader, 0):
        #we are doing this to do batch tokenization so we have 
        # better gpu speed
        text = data['text']
        text = [[x] for x in text]
        inputs = tokenizer(
            text,
            None,
            add_special_tokens=True,
            max_length=MAX_LEN,
            padding = 'longest',
            return_token_type_ids=True,
            truncation=True,
            return_tensors='pt',
            is_split_into_words=True
            )
        # used return_tensors='pt' to return pytorch tensors
        ids = inputs['input_ids'].to(device, dtype = torch.long)
        #print(ids)
        mask = inputs['attention_mask'].to(device, dtype = torch.long)
        targets = data['target'].to(device, dtype = torch.long)
        outputs = model(ids, mask)
        loss = loss_function(outputs, targets)
        tr_loss += loss.item()
        big_val, big_idx = torch.max(outputs.data, dim=1)
        n_correct += calcuate_accu(big_idx, targets)

        nb_tr_steps += 1
        nb_tr_examples+=targets.size(0)

        optimizer.zero_grad()
        loss.backward()
        # # When using GPU
        optimizer.step()

    print(f'The Total Accuracy for Epoch {epoch}: {(n_correct*100)/nb_tr_examples}')
    epoch_loss = tr_loss/nb_tr_steps
    epoch_accu = (n_correct*100)/nb_tr_examples
    print(f"Training Loss Epoch: {epoch_loss}")
    print(f"Training Accuracy Epoch: {epoch_accu}")

    return 


from sklearn.metrics import  precision_score, recall_score, f1_score
import numpy as np

def tensor2numpy(t):
    #print(t)
    l = []
    for tens in t:
        for val in tens:
            l.append(int(val.cpu().numpy()))
    a = np.array(l)
    return a


def valid(model, testing_loader):
    model.eval()
    n_correct = 0; n_wrong = 0; total = 0
    tr_loss = 0
    nb_tr_examples = 0
    nb_tr_steps  = 0
    output_list = []
    target_list = []
    with torch.no_grad():
        for _, data in enumerate(testing_loader, 0):
            #we are doing this to do batch tokenization so we have 
            # better gpu speed
            text = data['text']
            text = [[x] for x in text]
            inputs = tokenizer(
                text,
                None,
                add_special_tokens=True,
                max_length=MAX_LEN,
                padding = 'longest',
                return_token_type_ids=True,
                truncation=True,
                return_tensors='pt',
                is_split_into_words=True
                )
            # used return_tensors='pt' to return pytorch tensors
            ids = inputs['input_ids'].to(device, dtype = torch.long)
            #print(ids)
            mask = inputs['attention_mask'].to(device, dtype = torch.long)
            targets = data['target'].to(device, dtype = torch.long)
            target_list.append(targets)
            outputs = model(ids, mask).squeeze()
            big_val, big_idx = torch.max(outputs.data, dim=1)
            output_list.append(big_idx)
            loss = loss_function(outputs, targets)
            tr_loss += loss.item()
            n_correct += calcuate_accu(big_idx, targets)

            nb_tr_steps += 1
            nb_tr_examples+=targets.size(0)
    epoch_loss = tr_loss/nb_tr_steps
    epoch_accu = (n_correct*100)/nb_tr_examples
    o = tensor2numpy(output_list)
    t = tensor2numpy(target_list)
    
    f1 = f1_score(t,o,labels=[x for x in range(NO_OF_CLASS)],average=None)
    pre = precision_score(t,o,labels=[x for x in range(NO_OF_CLASS)],average=None)
    recall = recall_score(t,o,labels=[x for x in range(NO_OF_CLASS)],average=None)

    print(f"precision_score of this epoch is : {pre}")
    print(f"recall score of this epoch is : {recall}")
    print(f"f1 score of this epoch is: {f1}")
    print(f"Validation Loss Epoch: {epoch_loss}")
    print(f"Validation Accuracy Epoch: {epoch_accu}")
    
    return epoch_accu,f1, pre, recall

def main():
    # Creating the loss function and optimizer
    loss_function = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(params =  model.parameters(), lr=LEARNING_RATE)
    print("started")
    #EPOCHS = 1
    score_df  = pd.DataFrame()
    for epoch in range(EPOCHS):
        train(epoch)
        PATH = '/content/drive/MyDrive/NLP/temp_model/Refined1_bert_with9k_e' + str(epoch)+'.pt'
        torch.save(model.state_dict(), PATH)
        print('Now we test')
        acc,f1, pre, recall = valid(model, testing_loader)
        score_df = score_df.append({'epoch':epoch,'acc':acc,'f1':f1,'pre':pre,'recall':recall},ignore_index=True)
        PATH1 = '/content/drive/MyDrive/NLP/temp_model/Refined1_bert_with9k_score.csv'
        score_df.to_csv(PATH1)
        print('score_saved')
        print("Accuracy on test data = %0.2f%%" % acc)


    # Saving the files for re-use
    PATH = '/content/drive/MyDrive/NLP/torch_reBERT1/Refined_bert_0.pt'
    torch.save(model.state_dict(), PATH)
    tokenizer.save_vocabulary('/content/drive/MyDrive/NLP/torch_reBERT1')


if __name__ == "__main__":
    main()