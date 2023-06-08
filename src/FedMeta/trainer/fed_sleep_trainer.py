import logging

import torch
from torch import nn
import numpy as np

from fedml.core import ClientTrainer
import os
import time
from sklearn.metrics import confusion_matrix, auc, roc_curve, roc_auc_score, f1_score, classification_report


class FedMetaTrainer(ClientTrainer):
    
    def get_model_params(self):
        return self.model.cpu().state_dict()

    def set_model_params(self, model_parameters):
        self.model.load_state_dict(model_parameters)

    def train(self, train_data, device, args):
        model = self.model

        model.to(device)
        model.train()

        # train and update
        criterion = nn.CrossEntropyLoss().to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

        epoch_loss = []
        
        for epoch in range(args.epochs):
            batch_loss = []
            
            for batch_idx, x in enumerate(train_data):
                x, stage = x[0].to(device), x[1].to(device)
                
                optimizer.zero_grad()
                pred_value, _ = model(x)
                
                loss = criterion(pred_value, stage)
                loss.backward()
                optimizer.step()

                batch_loss.append(loss.item())
                
            epoch_loss.append(sum(batch_loss) / len(batch_loss))
            
            logging.info(
                "Client Index = {}\tEpoch: {}\tLoss: {:.6f}".format(self.id, epoch, sum(epoch_loss) / len(epoch_loss))
            )
        epoch_loss = np.array(epoch_loss)
        
        str_time = time.strftime("%y%m%d_%H%M%S")
        np.save(os.path.join(args.model_file_cache_folder, f"trainLoss_{str_time}"), epoch_loss)
        
        
    def test(self, test_data, device, args):
        
        model = self.model

        model.to(device)
        model.eval()
        
        criterion = nn.CrossEntropyLoss().to(device)
        
        val_pred = []
        val_real = []
        
        tot_correct, tot_samples = 0.0, 0.0
        tot_loss = 0
        
        with torch.no_grad():
            for batch_idx, x in enumerate(test_data):
                x, stage = x[0].to(device), x[1].to(device)
                
                pred_value, _ = model(x)
                
                loss = criterion(pred_value, stage)
                pred = torch.argmax(pred_value, dim=1)
                
                correct = torch.sum(pred == stage)
                
                val_pred.extend(pred.detach().cpu().numpy())
                val_real.extend(stage.detach().cpu().numpy())
                
                tot_correct += correct
                tot_samples += x.size(0)
                tot_loss += loss.item()
            
            running_loss = round(tot_loss / len(test_data), 3)
            acc = tot_correct / tot_samples
            
            try:
                f1score = round(f1_score(val_real, val_pred, average='macro'), 3)
                
            except:
                f1score = -1.0
            
            print("---------------------------------------------------")
            # print("AUC:", round(auc(fpr, tpr), 3))
            print("F1 Score:", round(f1score, 3))
            print(confusion_matrix(val_real, val_pred))
            print(classification_report(val_real, val_pred))
            print("---------------------------------------------------")
            
            logging.info(f"Client Index= {self.id}, Loss= {running_loss}, Accuracy= {acc}")
            
        running_loss = np.array(running_loss)
        acc = np.array(acc)
        
        str_time = time.strftime("%y%m%d_%H%M%S")
        np.save(os.path.join(args.model_file_cache_folder, f"testLoss_{str_time}"), running_loss)
        np.save(os.path.join(args.model_file_cache_folder, f"testAcc_{str_time}"), acc)
