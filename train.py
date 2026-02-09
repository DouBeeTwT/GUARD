from Scripts.model import MMM
from pytorch_lightning.callbacks import ModelCheckpoint
from Scripts.dataset import create_dataloader
from torchvision import transforms
import pytorch_lightning as pl
import argparse

parser = argparse.ArgumentParser(description="Train the model")
parser.add_argument('--epochs', type=int, default=100, help='Number of epochs to train')
parser.add_argument('--batch_size', type=int, default=32, help='Batch Size')
parser.add_argument('--lr', type=float, default=1e-3, help='Learning Rate')
parser.add_argument('--cls', type=int, default=2, help='Number of Classes (2 or 3)')
args = parser.parse_args()

alpha_dict = {2: [0.5, 0.5], 3: [0.12, 0.18, 0.7]}

checkpoint_callback = ModelCheckpoint(
    monitor='Val/ACC',
    dirpath='Checkpoints/',
    filename=f'best-model-class{args.cls}',
    save_top_k=1,
    mode='max',
    save_on_train_epoch_end=False,
)

transform_train_list = [
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
        transforms.ToTensor()]
transform_us = transforms.Compose(transform_train_list +
        [transforms.Normalize(mean=[0.278, 0.278, 0.278], std=[0.150, 0.150, 0.150])])
transform_dp = transforms.Compose(transform_train_list +
        [transforms.Normalize(mean=[0.272, 0.271, 0.271], std=[0.157, 0.155, 0.156])])

dataloader = create_dataloader(root=f"Dataset_Class{args.cls}", batch_size=16,
                               transform_us=transform_us, transform_dp=transform_dp)

model = MMM(lr=1e-4, mode = "US+DP+TX+MT+AA+CL",
            num_classes=args.cls, alpha=alpha_dict[args.cls],
            gamma=2, delta=0.5, device="cuda")

trainer = pl.Trainer(accelerator="gpu", devices=[0], max_epochs=args.epochs, callbacks=[checkpoint_callback])
trainer.fit(model, dataloader['train'], dataloader['test'])