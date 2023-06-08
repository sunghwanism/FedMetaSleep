import logging

import torch
from torch import nn
import numpy as np

from fedml.core import ClientTrainer
import os
import time


class FedDetectTrainer(ClientTrainer):
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
        time = time.strftime("%y%m%d_%H%M%S")
        np.save(os.path.join(args.model_file_cache_folder, f"learningLoss_{time}"), epoch_loss)
        
        
    def test(self, test_data, device, args):
        pass
