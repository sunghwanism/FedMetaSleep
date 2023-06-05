import os
import torch
from torch.utils.data import DataLoader, TensorDataset


import numpy as np
import pandas as pd




class applewatch:

    def __init__(self, database, length, meta_train_client_idx_lst=None,):
        
        self.x_data = []
        self.y_data = []
        
        if meta_train_client_idx_lst is None:
            meta_train_client_idx_lst = list(range(11, 30))
        
        for client in meta_train_client_idx_lst:
            PATH = os.path.join(database, f"c{client}_data.csv")
            data = pd.read_csv(PATH)
            
            for k in range(int(len(data)/length)):
                front_idx = int(k*length)
                post_idx = int((k+1)*length)
                
                temp = data[front_idx:post_idx]                
                x_move = temp["x_move"].to_numpy()
                y_move = temp["y_move"].to_numpy()
                z_move = temp["z_move"].to_numpy()
                HR = temp["heart_rate"].to_numpy()
                activity = temp["step_count"].to_numpy()
                stage = temp["psg_status"].to_numpy()[0].astype(int)
                # if stage == 5:
                #     stage = 4
                if stage in [1,2,3,4]:
                    stage = 1
                elif stage == 5:
                    stage = 2
                
                self.x_data.append(np.stack([x_move, y_move, z_move, HR, activity], axis=1))
                # self.x_data.append(np.stack([HR, activity], axis=1))
                self.y_data.append(stage)
        
        self.x_data = np.array(self.x_data)
        self.y_data = np.array(self.y_data)
        
        self.x_data = torch.FloatTensor(self.x_data)
        self.y_data = torch.FloatTensor(self.y_data).long()
        
        self.x_data = self.x_data.permute(0, 2, 1)
        
        print("x_data shape: ", self.x_data.shape)
        print("y_data shape: ", self.y_data.shape, "class_num", self.y_data.unique().shape[0])
        
    def __len__(self):
        return len(self.x_data)
    
    def __getitem__(self, idx):
        
        return self.x_data[idx], self.y_data[idx]


def create_train_val_loader(database, batch_size, length):
    
    print("#######################################")
    print("Train DataLoader")
    train_dataset = applewatch(database=database, length=length)
    
    print("#######################################")
    
    print("Validation DataLoader")
    valid_dataset = applewatch(database=database, length=length, meta_train_client_idx_lst=[30, 31])
    print("#######################################")
    
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    valid_dataloader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)

    return train_dataloader, valid_dataloader