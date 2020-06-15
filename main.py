import os
import argparse

import stream_train
from Logger import set_logger
from Modules import ResNet34, Centers, Memory
from init_train import load_or_init_train

if __name__ == '__main__':
    # run arguments
    parser = argparse.ArgumentParser(description="S2OSC")
    parser.add_argument("--dataset", type=str, default="c10", choices=["m", "fm", "c10", "cinic", "svhn"])
    parser.add_argument("--device", type=str, default="0", choices=["0", "1", "2", "3", "4", "5"])
    parser.add_argument("--train", type=bool, default=True, help="True to train, False to load pretrained files.")
    parser.add_argument("--init_epochs", type=int, default=3, help="Epochs to initially train Model_F.")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--pool_size", type=int, default=6000)
    parser.add_argument("--epochs_g", type=int, default=30, help="Epochs to train Model_G.")
    parser.add_argument("--K", type=int, default=300, help="Number of instances for each class stores in the memory.")
    args = parser.parse_args()

    # set cuda device
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ['CUDA_VISIBLE_DEVICES'] = args.device

    # set logger
    logger, log_folder = set_logger(name=f"{args.dataset}")
    logger.info("--------    Setting    --------")
    logger.info(f"dataset       {args.dataset}")
    logger.info(f"device        cuda:{args.device}")
    logger.info(f"train         {args.train}")
    logger.info(f"init_epochs   {args.init_epochs}")
    logger.info(f"batch_size    {args.batch_size}")
    logger.info(f"pool_size     {args.pool_size}")
    logger.info(f"epochs_g      {args.epochs_g}")
    logger.info(f"K             {args.K}")

    # initial train
    n_out = 10
    modelF = ResNet34(n_out=n_out).cuda()
    memory = Memory(K=300)
    centers = Centers(rate_old=0.8, n_centers=n_out)
    load_or_init_train(logger=logger, dataset_name=args.dataset, train=args.train, epochs=args.init_epochs, n_out=n_out,
                       model=modelF, memory=memory, centers=centers, running_path=log_folder)

    # stream train
    stream = stream_train.Stream(logger=logger, log_folder=log_folder, dataset_name=args.dataset,
                                 model_f=modelF, centers=centers, memory=memory,
                                 batch_size=args.batch_size, pool_size=args.pool_size,
                                 epochs_g=args.epochs_g, K=args.K)
    stream.train()
