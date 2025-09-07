from argparse import ArgumentParser
from pytorch_lightning.loggers import CSVLogger

import pytorch_lightning as pl
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.callbacks import ModelCheckpoint

from datamodules import INTERACTIONDataModule
from model import RuleNet

import torch
#torch.use_deterministic_algorithms(True)
   

if __name__ == '__main__':
    pl.seed_everything(1024, workers=True)

    parser = ArgumentParser()
    parser.add_argument('--root', type=str, required=True)
    parser.add_argument('--train_batch_size', type=int, default=1)
    parser.add_argument('--val_batch_size', type=int, default=1)
    parser.add_argument('--shuffle', type=bool, default=True)
    parser.add_argument('--num_workers', type=int, default=1)
    parser.add_argument('--pin_memory', type=bool, default=True)
    parser.add_argument('--persistent_workers', type=bool, default=True)
    parser.add_argument('--flip_p', type=float, default=0.5)
    parser.add_argument('--agent_occlusion_ratio', type=float, default=0.0)
    parser.add_argument('--lane_occlusion_ratio', type=float, default=0.2)
    parser.add_argument('--devices', type=int, required=True)
    parser.add_argument('--max_epochs', type=int, default=100)
    RuleNet.add_model_specific_args(parser)
    args = parser.parse_args()

    model = RuleNet(**vars(args))
    datamodule = INTERACTIONDataModule(**vars(args))
    model_checkpoint = ModelCheckpoint(monitor='val_minJointFDE', save_top_k=3, mode='min')
    lr_monitor = LearningRateMonitor(logging_interval='epoch')
    logger = CSVLogger("logs", name="loss.csv")
    trainer = pl.Trainer(devices=args.devices, accelerator='gpu', callbacks=[model_checkpoint, lr_monitor], max_epochs=args.max_epochs, logger=logger)
    trainer.fit(model, datamodule)
