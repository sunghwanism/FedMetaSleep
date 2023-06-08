"""
Copyright (c) Simon Schug
All rights reserved.

MIT License

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the “Software”), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""
import torch

from utils import config
from sklearn.metrics import confusion_matrix, auc, roc_curve, roc_auc_score, f1_score, classification_report

@torch.no_grad()
def accuracy(model, dataloader, device):
    """
    Compute accuracy on a given dataloader.
    """
    # Prepare the model for testing
    model.train(mode=False)
    tot_correct, tot_samples = 0.0, 0.0

    # Count correct predictions for all data points in test set
    val_pred = []
    val_real = []
    
    for x_batch, y_batch in dataloader:
        x_batch, y_batch = x_batch.to(device), y_batch.to(device)
        output, _ = model(x_batch)
        pred = torch.argmax(output, dim=1)
        correct = torch.sum(pred == y_batch)
        
        val_pred.extend(pred.detach().cpu().numpy())
        val_real.extend(y_batch.detach().cpu().numpy())
        
        tot_correct += correct
        tot_samples += x_batch.size(0)
    
    f1score = f1_score(val_real, val_pred, average='macro')
        
    # fpr, tpr, thresholds = roc_curve(val_pred, val_real, pos_label=2)
    print("---------------------------------------------------")
    # print("AUC:", round(auc(fpr, tpr), 3))
    print("F1 Score:", round(f1score, 3))
    print(confusion_matrix(val_real, val_pred))
    print(classification_report(val_real, val_pred))
    print("---------------------------------------------------")
    
    return tot_correct / tot_samples, f1score


def loss(model, dataloader, loss_function, device):
    """
    Compute the loss on a given dataloader.

    Args:
        loss_function: energy.EnergyFunction to compute the loss with
        dataloader: torch.utils.data.Dataloader
        model:models.NeuralNetwork model
        hyperparams: hyperparameters wrapped in meta.HyperparameterDict

    Returns:
        Loss summed over the dataloader
    """
    # Prepare the model for testing
    model.train(mode=False)
    loss_value = 0.0

    for x_batch, y_batch in dataloader:
        x_batch, y_batch = x_batch.to(device), y_batch.to(device)
        output, state = model(x_batch)
        loss_value += loss_function(output, y_batch, model, state)

    return loss_value
