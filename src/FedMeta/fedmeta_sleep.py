import sys
sys.path.append("./")

from fedml import fedml
from data.data_loader import load_data
from fedml import FedMLRunner
from model.depthwiseNet import DepthNet
from trainer.fedmeta_aggregator import FedMeta_aggregator
from trainer.fed_sleep_trainer import FedMetaTrainer

import torch

from fedml.arguments import Arguments, add_args
from fedml.constants import FEDML_TRAINING_PLATFORM_CROSS_SILO

_global_training_type = FEDML_TRAINING_PLATFORM_CROSS_SILO

def get_args():

    fedml._global_training_type = FEDML_TRAINING_PLATFORM_CROSS_SILO

    cmd_args = add_args()
    args = Arguments(cmd_args, training_type=_global_training_type,)
    # args.get_default_yaml_config(cmd_args, training_type=_global_training_type)
    args.load_yaml_config("./config/fedml_config.yaml")
    
    return args


if __name__ == "__main__":
    # init FedML framework
    args = get_args()
    args = fedml.init(args=args)

    # init device
    device = fedml.device.get_device(args)

    # load data
    dataset, output_dim = load_data(args, length=30)

    # load model
    model = DepthNet(lengths=30, patch_size=30, in_chans=2, embed_dim=256, norm_layer=None, output_dim=args.class_num)
        
    # pre_trained = torch.load("../Metamodel/log/best_model_t1t2.pt", map_location=device)
    # model.load_state_dict(pre_trained)

    # create trainer
    trainer = FedMetaTrainer(model, args)
    aggregator = FedMeta_aggregator(model, args)

    # start training
    fedml_runner = FedMLRunner(args, device, dataset, model, trainer, aggregator)
    fedml_runner.run()
