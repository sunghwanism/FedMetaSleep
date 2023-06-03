import os
import torch
from torch.utils.data import DataLoader, TensorDataset


import numpy as np
import pandas as pd




class applewatch:

    def __init__(self, database, batch_size, meta_train_client_idx_lst=None):
        
        self.x_data = []
        self.y_data = []
        
        if meta_train_client_idx_lst is None:
            meta_train_client_idx_lst = list(range(11, 30))
        
        for client in meta_train_client_idx_lst:
            PATH = os.path.join(database, f"c{client}_data.csv")
            data = pd.read_csv(PATH)
            
            HR = data["heart_rate"]
            activity = data["step_count"]
            stage = data["psg_status"].astype(int)
        
            self.x_data.append(np.stack([HR, activity], axis=1))
            self.y_data.append(np.array(stage))
            
        print("Data loaded")
        
    def __len__(self):
        return len(self.x_data)
    
    def __getitem__(self, idx):
        
        return torch.FloatTensor(self.x_data[idx]), torch.Tensor(self.y_data[idx]).long()


def create_train_val_loader(database, batch_size, ):
    
    
    train_dataset = applewatch(database=database, batch_size=batch_size)
    valid_dataset = applewatch(database=database, batch_size=batch_size, meta_train_client_idx_lst=[30, 31])
    
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    valid_dataloader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)

    return train_dataloader, valid_dataloader