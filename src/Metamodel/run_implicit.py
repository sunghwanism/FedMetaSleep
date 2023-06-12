"""
Copyright (c) Simon Schug
All rights reserved.

MIT License

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the “Software”), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""
import argparse
import json
import logging
import os
import sys
import time

sys.path.append("./")

import torch
from ray import tune


from data.dataloader import create_train_val_loader
import energy
import meta
import models
import train
import utils

from utils import config

from models.depthwiseNet import DepthNet


def run_implicit(raytune=False):
        
    # Initialize seed if specified (might slow down the model)
    seed = 1000
    torch.manual_seed(seed)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)
    
    batch_size = 512
    database = "../../data/new"

    # Create the training, validation and test dataloader
    train_loader, valid_loader = create_train_val_loader(database, batch_size, length=30)

    # Initialise the model
    # NOTE: Hard-coded output_dim as all datasets considered so far have 10 outputs
    # model = models.LeNet(output_dim=10).to(device)
    model = DepthNet(lengths=30, patch_size=30, in_chans=2, embed_dim=256, norm_layer=None, output_dim=3).to(device)
    
    best_f1 = 0
    
    # Initialise the implicit gradient approximation method
    ############################################################
    # Configure
    meta_method = "t1t2"
    cg_steps = 100
    
    nsa_steps = 500
    nsa_alpha = 0.0003
    
    init_l2 = 1e-02
    inner_init = "fixed_seed"
    optimizer_outer_type = "adam"
    lr_outer = 0.00001
    lr_inner = 0.00001
    
    steps_inner = 2000
    steps_outer = 100
    
    optimizer_inner_type = "sgd_nesterov"
    ############################################################
    if meta_method == "cg":
        implicit_gradient = meta.ConjugateGradient(cg_steps)
        
    elif meta_method == "nsa":
        implicit_gradient = meta.NeumannSeries(nsa_steps, nsa_alpha)
        
    elif meta_method == "t1t2":
        implicit_gradient = meta.T1T2()
        
    else:
        raise ValueError("Implicit gradient approximation method \"{}\" undefined".format(meta_method))
    

    # Initialise the hyperparameters
    hyperparams = meta.HyperparameterDict({"l2": [torch.full_like(p, init_l2) for p in model.parameters()]}).to(device)

    # Initialise meta model wrapping hyperparams and main model
    meta_model = meta.Hyperopt(model, hyperparams, inner_init=inner_init, nonnegative_keys={"l2"})

    # Initialise the energy functions
    inner_loss_function = energy.CrossEntropy() + energy.L2Regularizer()
    outer_loss_function = energy.CrossEntropy()

    # Initialise the outer-level optimizer
    optimizer_outer = utils.create_optimizer(optimizer_outer_type, hyperparams.parameters(), {"lr": lr_outer})

    results = {
        "test_acc": torch.zeros(steps_outer + 1),
        "test_loss": torch.zeros(steps_outer + 1),
        "train_acc": torch.zeros(steps_outer + 1),
        "train_loss": torch.zeros(steps_outer + 1),
        "valid_acc": torch.zeros(steps_outer + 1),
        "valid_loss": torch.zeros(steps_outer + 1),
    }

    # NOTE: There is one additional meta iteration which does not invoke a meta update:
    # +1 to evaluate the validation accuracy after the last outer step
    for step_outer in range(steps_outer + 1):
        # Initialise the model parameters
        meta_model.reset_parameters()                       

        # Initialise the inner-level optimizer
        optimizer_inner = utils.create_optimizer(optimizer_inner_type, model.parameters(), {"lr": lr_inner})

        # Inner-loop training
        train.train_augmented(
            meta_model, steps_inner, optimizer_inner, train_loader, inner_loss_function, verbose=not raytune,
        )

        if step_outer < steps_inner:
            print(step_outer)

            # HACK: Put all data in a single batch to save memory when backpropagating through the loss
            def dataloader_to_tensor(dataloader):
                x = torch.stack(list(zip(*list(dataloader.dataset)))[0])
                y = torch.tensor(list(zip(*list(dataloader.dataset)))[1])

                return (x, y)

            # Compute the inner and outer-loss
            inner_loss = train.loss(meta_model, [dataloader_to_tensor(train_loader)], inner_loss_function, device)
            outer_loss = train.loss(meta_model, [dataloader_to_tensor(valid_loader)], outer_loss_function, device)

            # Compute the indirect gradient wrt hyperparameters
            indirect_hypergrad = implicit_gradient.hypergrad(
                inner_loss, outer_loss, list(model.parameters()), list(hyperparams.parameters())
            )

            # Apply the outer optimisation step
            # NOTE: We assume that direct_hyper_grad == 0, i.e. the outer-loss does not depend on the hyperparams
            optimizer_outer.zero_grad()
            for hp, hp_grad in zip(hyperparams.parameters(), indirect_hypergrad):
                hp.grad = hp_grad

            optimizer_outer.step()

            # Enforce non-negativity through projected GD for selected hyperparameters
            meta_model.enforce_nonnegativity()

        # Testing
        with torch.no_grad():
            print("<< Train Set >>")
            train_acc, train_f1 = train.accuracy(model, train_loader, device)
            train_loss = train.loss(model, train_loader, outer_loss_function, device)
            
            print("<< Validation Set >>")
            valid_acc, valid_f1 = train.accuracy(model, valid_loader, device)
            valid_loss = train.loss(model, valid_loader, outer_loss_function, device)
            
            if best_f1 < valid_f1:
                best_f1 = valid_f1
                torch.save(model.state_dict(), os.path.join("log",f"best_model_{meta_method}.pt"))

        # Logging
        if raytune:
            tune.report(**{
                "train_acc": train_acc.item(),
                "train_loss": train_loss.item(),
                "valid_acc": valid_acc.item(),
                "valid_loss": valid_loss.item(),
            })
        else:
            print((
                "step_outer: {}/{}\t train_acc: {:4f} \t valid_acc: {:4f}".format(
                    step_outer, steps_outer, train_acc, valid_acc, 
                )
            ))
            print((
                "step_outer: {}/{}\t train_loss: {:4f} \t valid_loss: {:4f}".format(
                    step_outer, steps_outer, train_loss, valid_loss, 
                )
            ))
            logging.info(
                "step_outer: {}/{}\t train_acc: {:4f} \t valid_acc: {:4f}".format(
                    step_outer, steps_outer, train_acc, valid_acc, 
                )
            )
            logging.info(
                "step_outer: {}/{}\t train_loss: {:4f} \t valid_loss: {:4f}".format(
                    step_outer, steps_outer, train_loss, valid_loss, 
                )
            )


            results["train_acc"][step_outer] = train_acc
            results["train_loss"][step_outer] = train_loss
            results["valid_acc"][step_outer] = valid_acc
            results["valid_loss"][step_outer] = valid_loss

            # config.writer.add_scalars('accuracy', {'train': train_acc, 'valid': valid_acc}, step_outer)
            # config.writer.add_scalars('loss', {'train': train_loss, 'valid': valid_loss}, step_outer)

            # for name, p in model.named_parameters():
            #     config.writer.add_histogram('parameter/{}'.format(name), p.view(-1), step_outer)

            # for name, p in hyperparams.named_parameters():
            #     config.writer.add_histogram('hyperparameter/{}'.format(name), p.view(-1), step_outer)

    # Final Testing
    logging.info("Final training on full dataset (train + valid)")
    print(f"Best f1score", best_f1)
    
    return results, model, hyperparams

    # # Concatenate the full dataset (train + valid)
    # full_train_loader = data.create_multitask_loader([train_loader.dataset, valid_loader.dataset], cfg["batch_size"])

    # # Initialise the model parameters
    # meta_model.reset_parameters()

    # # Initialise the inner-level optimizer
    # optimizer_inner = utils.create_optimizer(
    #     cfg["optimizer_inner"], model.parameters(), {"lr": cfg["lr_inner"]}
    # )

    # # Inner-loop training
    # train.train_augmented(
    #     meta_model, cfg["steps_inner"], optimizer_inner, full_train_loader, inner_loss_function, verbose=not raytune
    # )

    # # Final testing
    # with torch.no_grad():
    #     train_acc_full = train.accuracy(meta_model, full_train_loader)
    #     train_loss_full = train.loss(meta_model, full_train_loader, outer_loss_function)

    # if raytune:
    #     tune.report(**{
    #         "train_acc": train_acc.item(),
    #         "train_loss": train_loss.item(),
    #         "valid_acc": valid_acc.item(),
    #         "valid_loss": valid_loss.item(),
    #         "train_acc_full": train_acc_full.item(),
    #         "train_loss_full": train_loss_full.item(),
    #     })

    # else:
    #     results["train_acc_full"] = train_acc_full
    #     results["train_loss_full"] = train_loss_full

    #     logging.info("train_acc_full: {:4f} ".format(train_acc_full))
    #     logging.info("train_loss_full: {:4f} ".format(train_loss_full))

        # return results, model, hyperparams


if __name__ == '__main__':
    # Load configuration (Priority: 1 User, 2 Random, 3 Default)
    
    results, model, hyperparams = run_implicit(raytune=False)
    LOG_DIR = "./log"

    # Save the configuration as json
    # utils.save_dict_as_json(cfg, run_id, config.LOG_DIR)

    # Store results, configuration and model state as pickle
    # results['cfg'], results['model'], results['hyperparameter'] = cfg, model.state_dict(), hyperparams.state_dict()
    results['model'], results['hyperparameter'] = model.state_dict(), hyperparams.state_dict()

    # Zip the tensorboard logging results and remove the folder to save space
    # config.writer.close()
    # path_tensorboard = os.path.join(LOG_DIR, "_tensorboard")
    # utils.zip_and_remove((path_tensorboard))