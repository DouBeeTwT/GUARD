from torchvision import transforms
import torch
import pytorch_lightning as pl
from model import MMM
from dataset import create_dataloader, transform_us, transform_dp
import config
import os
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import WandbLogger

torch.set_float32_matmul_precision('medium')

if __name__ == "__main__":
    logger = WandbLogger(project=config.PROJECT_NAME,
                         name=config.MODEL_NAME)
    
    dataloader = create_dataloader(
        root=config.DATA_DIR,
        batch_size=config.BATCH_SIZE,
        transform_us=transform_us,
        transform_dp=transform_dp,
    )

    model = MMM(lr = config.LEARNING_RATE,
                single_decoder = config.SINGLE_DECODER,
                mode = config.MODE,
                num_classes= config.NUM_CLASSES,
                alpha=config.ALPHA,
                gamma=config.GAMMA,
                delta = config.DELTA)

    trainer = pl.Trainer(
        logger = logger,
        accelerator=config.ACCELERATOR,
        devices=config.DEVICES,
        max_epochs=config.NUM_EPOCHS,
        callbacks=[
            ModelCheckpoint(
            monitor = config.MONITOR,
            mode = config.MONITOR_MODE,
            save_top_k = 1,
            dirpath = config.CHECKPOINT_DIR,
            filename = config.CHECKPOINT_NAME
            ), 
            EarlyStopping(
                monitor = config.MONITOR,
                mode = config.MONITOR_MODE,
                patience = config.EARLYSTOP_PATIENCE)]
    )

    trainer.fit(model, dataloader["train"], dataloader["test"])
