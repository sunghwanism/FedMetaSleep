import os
import sys
import time

sys.path.append("./")

import torch
import torch.nn as nn

from data.dataloader import create_train_val_loader

import models
from models.depthwiseNet import DepthNet

import train
import utils
import energy

from sklearn.metrics import confusion_matrix, auc, roc_curve, roc_auc_score, f1_score



def run_model():
    
    # Initialize seed if specified (might slow down the model)
    seed = 1
    torch.manual_seed(seed)
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    print("Device: ", device)
    
    epochs = 500
    batch_size = 512
    database = "../../data/padding"

    # Create the training, validation and test dataloader
    train_loader, valid_loader = create_train_val_loader(database, batch_size, length=30)
    

    # Initialise the model
    # NOTE: Hard-coded output_dim as all datasets considered so far have 10 outputs
    # model = models.LeNet(output_dim=5).to(device)
    model = DepthNet(lengths=30, patch_size=30, in_chans=5, embed_dim=256, norm_layer=None, output_dim=3).to(device)
    
    
    # Initialise the implicit gradient approximation method
    ############################################################
    
    lr = 0.0001
    # optimizer_outer = utils.create_optimizer(optimizer_outer, model.parameters(), {"lr": lr})
    optimizer_outer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=0.001)
    criterion = nn.CrossEntropyLoss().to(device)
    
    best_acc = 0
    best_f1 = 0
    # best_auc = 0
    for epoch in range(epochs):
        epoch += 1
        model.train()
        running_loss = 0.0
        
        for data in train_loader:            
            x_data, stage = data[0].to(device), data[1].to(device)
            
            optimizer_outer.zero_grad()
            # pred_value, _ = model(x_data)
            pred_value, _ = model(x_data)
            loss = criterion(pred_value, stage)
            loss.backward()
            optimizer_outer.step()
            
            running_loss += loss.item()
            
        running_loss /= len(train_loader)
        
        print(f"(Train) Epoch: {epoch}, Loss: {round(running_loss, 3)}")
        
        if epoch % 1 == 0:
            print("<< Validation >>")
            correct = 0
            valid_loss = 0.0
            
            model.eval()
            val_pred = []
            val_real = []
            with torch.no_grad():
                for data in valid_loader:
                    x_data, stage = data[0].to(device), data[1].to(device)
                    
                    # pred_value, _ = model(x_data)
                    pred_value, _ = model(x_data)
                    pred_class = torch.argmax(pred_value, dim=1)
                    loss = criterion(pred_value, stage)
                    
                    valid_loss += loss.item()
                    correct += (pred_class == stage).sum().item()
                    
                    val_pred.extend(pred_class.detach().cpu().numpy())
                    val_real.extend(stage.detach().cpu().numpy())
                
                acc = correct / len(valid_loader.dataset)
                f1 = f1_score(val_real, val_pred, average="macro")
                fpr, tpr, thresholds = roc_curve(val_pred, val_real, pos_label=1)
                
                if best_acc < acc:
                    best_acc = acc
                    # torch.save(model.state_dict(), "./model/depthwiseNet.pth")
                    
                if best_f1 < f1:
                    best_f1 = f1
                    torch.save(model.state_dict(), "./log/Base_3class.pth")
                
                # roc_score = roc_auc_score(val_real, val_pred, multi_class="ovr", average="macro")
                print(f"(Valid) Epoch: {epoch}, Loss: {round(valid_loss, 3)}, Acc: {round(acc, 3)}, F1: {round(f1, 3)}")
                # print(f"(Valid) Epoch: {epoch}, Loss: {round(valid_loss, 3)}, Accuracy: {round(acc, 3)},") # AUC: {round(roc_score, 3)}")
                
                print(confusion_matrix(val_real, val_pred))
                
    print("Best ACC: ", best_acc)
    print("Best f1: ", best_f1)
        
        
def main():
    run_model()
    
if __name__ == "__main__":
    main()