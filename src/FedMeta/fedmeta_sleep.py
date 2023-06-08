import sys
sys.path.append("./")


from fedml import fedml
from data.data_loader import load_data
from fedml import FedMLRunner
from model.depthwiseNet import DepthNet
from trainer.fedmeta_aggregator import FedMeta_aggregator
from trainer.fed_sleep_trainer import FedMetaTrainer

if __name__ == "__main__":
    # init FedML framework
    args = fedml.init()

    # init device
    device = fedml.device.get_device(args)

    # load data
    dataset, output_dim = load_data(args, length=30)

    # load model
    model = DepthNet(lengths=30, patch_size=30, in_chans=5, embed_dim=256, norm_layer=None, output_dim=args.class_num)

    # create trainer
    trainer = FedMetaTrainer(model, args)
    aggregator = FedMeta_aggregator(model, args)

    # start training
    fedml_runner = FedMLRunner(args, device, dataset, model, trainer, aggregator)
    fedml_runner.run()