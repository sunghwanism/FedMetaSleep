import logging
import os
import urllib.request
import zipfile

import numpy as np
import pandas as pd
import torch


def load_data(args):

    train_data_global = list()
    test_data_global = list()
    train_data_local_dict = dict.fromkeys(range(args.client_num_in_total))
    test_data_local_dict = dict.fromkeys(range(args.client_num_in_total))
    train_data_local_num_dict = dict.fromkeys(range(args.client_num_in_total))
    train_data_num = 0
    test_data_num = 0

    if args.rank == 0:
        for i in range(args.client_num_in_total):
            PATH = os.path.join(args.database, f"c{i+1}_data.csv")
            data = pd.read_csv(PATH)
            
            HR = data["heart_rate"]
            activity = data["step_count"]
            stage = data["psg_status"].astype(int)
            
            data = np.stack([HR, activity, stage], axis=1)
            
            train_data = None
            test_data = None
            
            train_data_local_dict[i] = torch.utils.data.DataLoader(
                train_data, batch_size=args.batch_size, shuffle=False, num_workers=2
            )
            
            test_data_local_dict[i] = torch.utils.data.DataLoader(
                test_data, batch_size=args.batch_size, shuffle=False, num_workers=2
            )
            train_data_local_num_dict[i] = len(train_data_local_dict[i])
            train_data_num += train_data_local_num_dict[i]

    else:
        PATH = os.path.join(args.database, f"c{args.rank}_data.csv")
        data = pd.read_csv(PATH)
            
        HR = data["heart_rate"]
        activity = data["step_count"]
        stage = data["psg_status"].astype(int)
        
        data = np.stack([HR, activity, stage], axis=1)
        
        train_data = None
        test_data = None           

        train_data_local_dict[args.rank - 1] = torch.utils.data.DataLoader(
            train_data, batch_size=args.batch_size, shuffle=True, num_workers=2
        )
        
        test_data_local_dict[args.rank - 1] = torch.utils.data.DataLoader(
            test_data, batch_size=args.batch_size, shuffle=False, num_workers=2
        )
        
        train_data_local_num_dict[args.rank - 1] = len(
            train_data_local_dict[args.rank - 1]
        )
        train_data_num += train_data_local_num_dict[args.rank - 1]

    class_num = len(stage.unique())
    dataset = [
        train_data_num,
        test_data_num,
        train_data_global,
        test_data_global,
        train_data_local_num_dict,
        train_data_local_dict,
        test_data_local_dict,
        class_num,
    ]
    return dataset, class_num
