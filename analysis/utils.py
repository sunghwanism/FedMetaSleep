import os
import sys
sys.path.append("../")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import seaborn as sns

from natsort import natsorted

import torch
import torch.nn as nn

import random

from src.Metamodel.models.depthwiseNet import DepthNet
from src.Metamodel.data.dataloader import create_train_val_loader
from torch.utils.data import random_split, DataLoader

from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import confusion_matrix, auc, roc_auc_score, f1_score, classification_report
from torchmetrics.classification import MulticlassAUROC


def get_round_loss_score(DATABASE, MODELBASE, basemodel, device, client, weightedType="macro", all_round=True):
    
    generator = torch.Generator()
    generator.manual_seed(0)
    
    client = [client]
    train_set, validadtion_set = create_train_val_loader(DATABASE, batch_size=256, 
                                                         length=30, meta_train_client_idx_lst=client, FLtrain=True)
    client_train_set, client_test_set = random_split(train_set,
                                                     [int(len(train_set) * 0.8), 
                                                      len(train_set) - int(len(train_set) * 0.8)], 
                                                     generator=generator)


    client_train_loader = DataLoader(client_train_set, batch_size=256, shuffle=False)
    client_test_loader = DataLoader(client_test_set, batch_size=256, shuffle=False)
    valid_loader = DataLoader(validadtion_set, batch_size=256, shuffle=False)
    
    models = natsorted(os.listdir(MODELBASE))
    auroc = MulticlassAUROC(num_classes=3, average=weightedType)
    criterion = nn.CrossEntropyLoss().to(device)
        
    # train_result_dic = {"loss": [], "acc": [], "f1": [], "auc_0": [], "auc_1": [], "auc_2": [], "confusion_matrix": []}
    # test_result_dic = {"loss": [], "acc": [], "f1": [], "auc_0": [], "auc_1": [], "auc_2": [], "confusion_matrix": []}
    train_result_dic = {"loss": [], "acc": [], "f1": [], "auc": [], "confusion_matrix": []}
    test_result_dic = {"loss": [], "acc": [], "f1": [], "auc": [], "confusion_matrix": []}
    
    if all_round:
        for round_model in models:

            savemodel = torch.load(os.path.join(MODELBASE, round_model), map_location=device)
            basemodel.load_state_dict(savemodel)
            basemodel.eval()
            
            train_correct = 0
            train_loss = 0
            
            test_correct = 0
            test_loss = 0.0

            train_pred = []
            train_real = []
            train_proba = []
            test_pred = []
            test_real = []
            test_proba = []
            
            with torch.no_grad():
                for data in client_train_loader:
                    x_data, stage = data[0].to(device), data[1].to(device)
                    
                    pred_value, _ = basemodel(x_data)
                    pred_class = torch.argmax(pred_value, dim=1)
                    pred_proba = torch.sigmoid(pred_value)
                    
                    loss = criterion(pred_value, stage)
                    
                    train_loss += loss.item()
                    train_correct += (pred_class == stage).sum().item()
                    
                    train_pred.extend(pred_class.detach().cpu().numpy())
                    train_real.extend(stage.detach().cpu().numpy())
                    train_proba.extend(pred_proba.detach().cpu().numpy())
                    
                
                # binarizer = LabelBinarizer().fit(train_real)
                # binarized_train_pred = binarizer.transform(train_pred)
                            
                # train_pred_0 = binarized_train_pred[:, 0]
                # train_pred_1 = binarized_train_pred[:, 1]
                # train_pred_2 = binarized_train_pred[:, 2]
                
                # roc_auc_ovr_0 = roc_auc_score(train_pred_0, train_proba, multi_class="ovr", average=weightedType)
                # roc_auc_ovr_1 = roc_auc_score(train_pred_1, train_proba, multi_class="ovr", average=weightedType)
                # roc_auc_ovr_2 = roc_auc_score(train_pred_2, train_proba, multi_class="ovr", average=weightedType)
                
                auc = auroc(torch.tensor(train_proba), torch.tensor(train_real))

                acc = train_correct / len(client_train_loader.dataset)
                f1score = f1_score(train_real, train_pred, average=weightedType)    
                
                train_result_dic["loss"].append(train_loss)
                train_result_dic["acc"].append(acc)
                train_result_dic["f1"].append(f1score)
                train_result_dic["auc"].append(auc.numpy())
                train_result_dic["confusion_matrix"].append(confusion_matrix(train_pred, train_real))
                # train_result_dic["auc_0"].append(roc_auc_ovr_0)
                # train_result_dic["auc_1"].append(roc_auc_ovr_1)
                # train_result_dic["auc_2"].append(roc_auc_ovr_2)
            

            with torch.no_grad():
                for data in client_test_loader:
                    x_data, stage = data[0].to(device), data[1].to(device)
                    
                    pred_value, _ = basemodel(x_data)
                    pred_class = torch.argmax(pred_value, dim=1)
                    pred_proba = torch.sigmoid(pred_value)
                    
                    loss = criterion(pred_value, stage)
                    
                    test_loss += loss.item()
                    test_correct += (pred_class == stage).sum().item()
                    
                    test_pred.extend(pred_class.detach().cpu().numpy())
                    test_real.extend(stage.detach().cpu().numpy())
                    test_proba.extend(pred_proba.detach().cpu().numpy())
                
                auc = auroc(torch.tensor(test_proba), torch.tensor(test_real))
                
                acc = test_correct / len(client_test_loader.dataset)
                f1score = f1_score(test_real, test_pred, average=weightedType)
                
                test_result_dic["loss"].append(test_loss)
                test_result_dic["acc"].append(acc)
                test_result_dic["f1"].append(f1score)
                test_result_dic["auc"].append(auc.numpy())
                test_result_dic["confusion_matrix"].append(confusion_matrix(test_pred, test_real))
                
    else:
        basemodel.eval()
        
        train_correct = 0
        train_loss = 0
        
        test_correct = 0
        test_loss = 0.0

        train_pred = []
        train_real = []
        train_proba = []
        test_pred = []
        test_real = []
        test_proba = []
        
        with torch.no_grad():
            for data in client_train_loader:
                x_data, stage = data[0].to(device), data[1].to(device)
                
                pred_value, _ = basemodel(x_data)
                pred_class = torch.argmax(pred_value, dim=1)
                pred_proba = torch.sigmoid(pred_value)
                
                loss = criterion(pred_value, stage)
                
                train_loss += loss.item()
                train_correct += (pred_class == stage).sum().item()
                
                train_pred.extend(pred_class.detach().cpu().numpy())
                train_real.extend(stage.detach().cpu().numpy())
                train_proba.extend(pred_proba.detach().cpu().numpy())
            
            auc = auroc(torch.tensor(train_proba), torch.tensor(train_real))

            acc = train_correct / len(client_train_loader.dataset)
            f1score = f1_score(train_real, train_pred, average=weightedType)
            
            train_result_dic["loss"].append(train_loss)
            train_result_dic["acc"].append(acc)
            train_result_dic["f1"].append(f1score)
            train_result_dic["auc"].append(auc.numpy())
            train_result_dic["confusion_matrix"].append(confusion_matrix(train_pred, train_real))

        with torch.no_grad():
            for data in client_test_loader:
                x_data, stage = data[0].to(device), data[1].to(device)
                
                pred_value, _ = basemodel(x_data)
                pred_class = torch.argmax(pred_value, dim=1)
                pred_proba = torch.sigmoid(pred_value)
                            
                loss = criterion(pred_value, stage)
                
                test_loss += loss.item()
                test_correct += (pred_class == stage).sum().item()
                
                test_pred.extend(pred_class.detach().cpu().numpy())
                test_real.extend(stage.detach().cpu().numpy())
                test_proba.extend(pred_proba.detach().cpu().numpy())
                
            auc = auroc(torch.tensor(test_proba), torch.tensor(test_real))
            
            acc = test_correct / len(client_test_loader.dataset)
            f1score = f1_score(test_real, test_pred, average=weightedType)
            
            test_result_dic["loss"].append(test_loss)
            test_result_dic["acc"].append(acc)
            test_result_dic["f1"].append(f1score)
            test_result_dic["auc"].append(auc.numpy())
            test_result_dic["confusion_matrix"].append(confusion_matrix(test_pred, test_real))
            
    return train_result_dic, test_result_dic


def train_one_epoch_output(DATABASE, MODELBASE, basemodel, device, client, round, weightedType="macro"):
    
    round_idx =  round -1
    temp_client_lst = [client]
    train_set, validadtion_set = create_train_val_loader(DATABASE, batch_size=256, 
                                                         length=30, meta_train_client_idx_lst=temp_client_lst, FLtrain=True)
    
    generator = torch.Generator()
    generator.manual_seed(0)
    
    client_train_set, client_test_set = random_split(train_set,
                                                     [int(len(train_set) * 0.8), 
                                                      len(train_set) - int(len(train_set) * 0.8)], 
                                                     generator=generator)


    client_train_loader = DataLoader(client_train_set, batch_size=256, shuffle=False)
    
    round_best_model = natsorted(os.listdir(MODELBASE))[round_idx]
    savemodel = torch.load(os.path.join(MODELBASE, round_best_model), map_location=device)
    basemodel.load_state_dict(savemodel)
    
    basemodel.train()
    
    optimizer_outer = torch.optim.Adam(basemodel.parameters(), lr=0.001, weight_decay=0.00001)
    criterion = nn.CrossEntropyLoss().to(device)
    

    for data in client_train_loader:            
        x_data, stage = data[0].to(device), data[1].to(device)
    
        optimizer_outer.zero_grad()
        # pred_value, _ = model(x_data)
        pred_value, _ = basemodel(x_data)
        loss = criterion(pred_value, stage)
        loss.backward()
        optimizer_outer.step()
        
    
    return get_round_loss_score(DATABASE, MODELBASE, basemodel, device, client, weightedType=weightedType, all_round=False)


class EarlyStopping:
    def __init__(self, patience=5):
        self.loss = np.inf
        self.patience = 0
        self.patience_limit = patience
        
    def step(self, loss):
        if self.loss > loss:
            self.loss = loss
            self.patience = 0
        else:
            self.patience += 1
    
    def is_stop(self):
        return self.patience >= self.patience_limit