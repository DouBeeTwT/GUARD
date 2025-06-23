import os
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import random
import pandas as pd
from torchvision import transforms

transform_train_list = [
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
        transforms.ToTensor(),
    ]
transform_us = transforms.Compose(
        transform_train_list
        + [transforms.Normalize(mean=[0.278, 0.278, 0.278], std=[0.150, 0.150, 0.150])]
    )
transform_dp = transforms.Compose(
        transform_train_list
        + [transforms.Normalize(mean=[0.272, 0.271, 0.271], std=[0.157, 0.155, 0.156])]
    )

class MultimodalDataset(Dataset):
    def __init__(
        self, us_dir, dp_dir, textes, mts, samples, transform_us=None, transform_dp=None, need_filename = False
    ):
        self.us_dir = us_dir
        self.dp_dir = dp_dir
        self.samples = samples
        self.text = textes
        self.mts = mts
        self.transform_us = transform_us
        self.transform_dp = transform_dp
        self.label_map = {"N0": 0, "N1-2": 1, "N3":2}
        self.need_filename = need_filename

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        us_image = Image.open(os.path.join(self.us_dir, self.samples[idx])).convert(
            "RGB"
        )
        dp_image = Image.open(os.path.join(self.dp_dir, self.samples[idx])).convert(
            "RGB"
        )
        us_image = self.transform_us(us_image)
        dp_image = self.transform_dp(dp_image)
        text = self.text[self.samples[idx]]
        mt = torch.FloatTensor(self.mts[self.samples[idx]])
        if self.need_filename:
            return (us_image, dp_image, text, mt, self.label_map[self.samples[idx].split("/")[0]], self.samples[idx])
        else:
            return (us_image, dp_image, text, mt, self.label_map[self.samples[idx].split("/")[0]])


def create_dataloader(
    root="Dataset", batch_size=32, transform_us=None, transform_dp=None, need_filename=False):
    us_dir = os.path.join(root, "ultrasound")
    dp_dir = os.path.join(root, "doppler")
    tx_dir = os.path.join(root, "echo_feature_text.csv")
    mt_dir = os.path.join(root, "matrix289.csv")

    random.seed(42)
    label_list = []
    for label in os.listdir(us_dir):
        f = [
            os.path.join(label, file_name)
            for file_name in os.listdir(os.path.join(us_dir, label))
        ]
        label_list.extend(f)
    random.shuffle(label_list)
    sample_train = label_list[: int(len(label_list) * 0.8)]
    sample_test = label_list[int(len(label_list) * 0.8) :]
    textes_df = pd.read_csv(tx_dir)
    textes = textes_df.set_index("Figure_pathway")["Text"].to_dict()
    mts_df = pd.read_csv(mt_dir)
    mts = (
        mts_df.set_index("Figure Pathway")
        .apply(lambda x: x.values.tolist(), axis=1)
        .to_dict()
    )

    dataset_train = MultimodalDataset(
        us_dir,
        dp_dir,
        samples=sample_train,
        textes=textes,
        mts=mts,
        transform_us=transform_us,
        transform_dp=transform_dp,
        need_filename = need_filename
    )
    transform_us_test = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.278, 0.278, 0.278], std=[0.150, 0.150, 0.150]),
        ]
    )
    transform_dp_test = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.272, 0.271, 0.271], std=[0.157, 0.155, 0.156]),
        ]
    )
    dataset_test = MultimodalDataset(
        us_dir,
        dp_dir,
        samples=sample_test,
        textes=textes,
        mts=mts,
        transform_us=transform_us_test,
        transform_dp=transform_dp_test,
        need_filename = need_filename
    )

    dataloader_train = DataLoader(dataset_train, batch_size=batch_size, shuffle=True)
    dataloader_test = DataLoader(dataset_test, batch_size=batch_size, shuffle=False)

    return {"train": dataloader_train, "test": dataloader_test}
