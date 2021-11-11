from torch.utils.data import Dataset
import pandas as pd
import torch

# define a custome dataloader

class MyDataset(Dataset):
    def __init__(self, dataframe):
        self.len = len(dataframe)
        self.data = dataframe
        #self.tokenizer = tokenizer
        #self.max_len = max_len
        
    def __getitem__(self, index):
        text = str(self.data.sw_Text[index])
        text = " ".join(text.split())
        print(self.data.Emotions[index])
        target = torch.tensor(self.data.Emotions[index].astype(int))
        return {'text' : text, 'target' : target}

    
    def __len__(self):
        return self.len