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
import random
import numpy as np
import torch.backends.cudnn as cudnn

# from pytorchtools import EarlyStopping
from utils.utils import EarlyStopping

from sklearn.metrics import confusion_matrix, auc, roc_auc_score, f1_score, classification_report
from torchmetrics.classification import MulticlassAUROC

torch.manual_seed(0)
torch.cuda.manual_seed(0)
torch.cuda.manual_seed_all(0)
np.random.seed(0)
cudnn.benchmark = False
cudnn.deterministic = True
random.seed(0)
np.random.seed(0)
torch.cuda.manual_seed_all(0)

import time
import datetime



def run_model(num=1):

    start = time.time()
    # Initialize seed if specified (might slow down the model)
    seed = 0 # Client 9
    # num = list(range(10,30))
    torch.manual_seed(seed)
    device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")
    print("Device: ", device)

    epochs = 50
    batch_size = 256
    database = "../../data/new"
    weightedType = "macro"

    # Create the training, validation and test dataloader
    #
    train_set, validation_set = create_train_val_loader(database, batch_size, length=30,
                                                        meta_train_client_idx_lst=[num], FLtrain=True)
    early_stop = EarlyStopping(patience=15)

    if type(num) == list:
        train = train_set
    else:
        train, test = torch.utils.data.random_split(train_set, [int(len(train_set)*0.8), len(train_set)-int(len(train_set)*0.8)], generator=torch.Generator().manual_seed(0))

    if type(num) == list:
        train_loader = torch.utils.data.DataLoader(train, batch_size=batch_size, shuffle=True)
        valid_loader = torch.utils.data.DataLoader(validation_set, batch_size=batch_size, shuffle=False)
    else:
        train_loader = torch.utils.data.DataLoader(train, batch_size=batch_size, shuffle=True)
        test_loader = torch.utils.data.DataLoader(test, batch_size=batch_size, shuffle=False)
        valid_loader = torch.utils.data.DataLoader(validation_set, batch_size=batch_size, shuffle=False)

    model = DepthNet(lengths=30, patch_size=30, in_chans=2, embed_dim=256, norm_layer=None, output_dim=3).to(device)

    lr = 0.0001
    optimizer_outer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=0.0001)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer=optimizer_outer, step_size=40, gamma=0.5, last_epoch=-1, verbose=False)
    criterion = nn.CrossEntropyLoss().to(device)

    best_acc = 0
    best_f1 = 0
    best_auc = 0
    auroc = MulticlassAUROC(num_classes=3, average=weightedType).to(device)

    finish = False
    early_epoch = 1
    train_result_dic = {"loss": [], "acc": [], "f1": [], "auc": [], "confusion_matrix": []}
    test_result_dic = {"loss": [], "acc": [], "f1": [], "auc": [], "confusion_matrix": []}

    for epoch in range(epochs):
        epoch += 1
        model.train()
        running_loss = 0.0
        correct = 0

        train_pred = []
        train_real = []
        train_proba = []
        
        for data in train_loader:            
            x_data, stage = data[0].to(device), data[1].to(device)
            
            pred_value, _ = model(x_data)
            pred = torch.argmax(pred_value, dim=1)
            pred_proba = torch.sigmoid(pred_value)

            correct += torch.sum(pred==stage).item()

            train_pred.extend(pred.detach().cpu().numpy())
            train_real.extend(stage.detach().cpu().numpy())
            train_proba.extend(pred_proba.detach().cpu().numpy())

            loss = criterion(pred_value, stage)

            optimizer_outer.zero_grad()
            loss.backward()
            optimizer_outer.step()
            
            running_loss += loss.item()
        
        auc = auroc(torch.tensor(train_proba), torch.tensor(train_real))
        acc = correct / len(train_loader.dataset)
        running_loss /= len(train_loader)
        f1score = f1_score(train_real, train_pred, average=weightedType)

        scheduler.step()
        
        print(f"(Train) Epoch: {epoch}, Loss: {round(running_loss, 3)}, AUC: {round(float(auc),3)}, ACC: {round(acc,3)}, F1score: {round(f1score, 3)}")

        train_result_dic["loss"].append(running_loss)
        train_result_dic["acc"].append(acc)
        train_result_dic["f1"].append(f1score)
        train_result_dic["auc"].append(auc.numpy())
        train_result_dic["confusion_matrix"].append(confusion_matrix(train_pred, train_real))
        
        if epoch % 1 == 0:
            print("<< Validation >>")

            correct = 0
            valid_loss = 0.0
            
            model.eval()

            val_pred = []
            val_real = []
            val_proba = []

            with torch.no_grad():
                for data in valid_loader:
                    x_data, stage = data[0].to(device), data[1].to(device)
                    
                    # pred_value, _ = model(x_data)
                    pred_value, _ = model(x_data)
                    pred = torch.argmax(pred_value, dim=1)
                    pred_proba = torch.sigmoid(pred_value)

                    correct += torch.sum(pred==stage).item()

                    val_pred.extend(pred.detach().cpu().numpy())
                    val_real.extend(stage.detach().cpu().numpy())
                    val_proba.extend(pred_proba.detach().cpu().numpy())

                    loss = criterion(pred_value, stage)
                    valid_loss += loss.item()
                
                auc = auroc(torch.tensor(val_proba), torch.tensor(val_real))
                acc = correct / len(valid_loader.dataset)
                valid_loss /= len(valid_loader)
                f1score = f1_score(val_real, val_pred, average=weightedType)            
                    
                if best_f1 < f1score:
                    best_f1 = f1score
                    best_epoch = epoch

                    if type(num) == list:
                        # torch.save(model.state_dict(), f"../singlelog/total_3class.pt")
                        pass
                    else:
                        # torch.save(model.state_dict(), f"../singlelog/c{num}_3class.pt")
                        pass

                print("################################################################################################")
                print(f"(Valid) Epoch: {epoch}, Loss: {round(valid_loss, 3)}, AUC: {round(float(auc),3)}, ACC: {round(acc,3)}, F1score: {round(f1score, 3)}")
                print("################################################################################################")


                test_result_dic["loss"].append(valid_loss)
                test_result_dic["acc"].append(acc)
                test_result_dic["f1"].append(f1score)
                test_result_dic["auc"].append(auc.numpy())
                test_result_dic["confusion_matrix"].append(confusion_matrix(val_pred, val_real))
            
        early_stop.step(f1score)
        
        if early_stop.is_stop():
            print("Early Stopping in Epoch", epoch)
            if type(num) == list:
                if finish != True:
                    # torch.save(model.state_dict(), f"../singlelog/total_3class_earlystop{epoch}.pt")
                    early_epoch = epoch
                    finish=True
            else:
                if finish != True:
                    # torch.save(model.state_dict(), f"../singlelog/c{num}_3class_earlystop{epoch}.pt")
                    finish=True
                    early_epoch = epoch
            # break
    print("Best f1: ", best_f1)

    end = time.time()
    sec = end - start
    print("Total Trining Time")
    result_time = str(datetime.timedelta(seconds=sec)).split(".")
    print(result_time[0])


        
def main():
    run_model()
    
if __name__ == "__main__":
    main()