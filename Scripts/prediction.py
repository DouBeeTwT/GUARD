from Scripts.model import MMM
import torch
import pandas as pd
from torchvision import transforms
from PIL import Image
import os

filename = "N3/0023.png"
device = "cuda:3"
ckpt_path = 'Checkpoints/best-model-class3.ckpt'

def predicrion(filename:str, ckpt_path:str, device:str):
    # Load Model
    model = MMM(lr=1e-4, mode = "US+DP+TX+MT+AA+CL",
                num_classes=3, alpha=[0.12, 0.18, 0.7],
                gamma=2, delta=0.5, device=device).eval()
    model.load_state_dict(
        torch.load(ckpt_path, map_location=device)["state_dict"]
        )
    # Load Data
    transform_us_test = transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.278, 0.278, 0.278], std=[0.150, 0.150, 0.150]),
            ]
        )
    x_us = transform_us_test(Image.open(f"Dataset_Class3/ultrasound/{filename}").convert("RGB"))

    transform_dp_test = transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.272, 0.271, 0.271], std=[0.157, 0.155, 0.156]),
            ]
        )
    x_dp = transform_dp_test(Image.open(f"Dataset_Class3/doppler/{filename}").convert("RGB"))

    textes_df = pd.read_csv("Dataset_Class3/echo_feature_text.csv")
    textes = textes_df.set_index("Figure_pathway")["Text"].to_dict()
    x_tx = textes[filename]

    mts_df = pd.read_csv("Dataset_Class3/matrix289.csv")
    mts = mts_df.set_index("Figure Pathway").apply(lambda x: x.values.tolist(), axis=1).to_dict()
    x_mt = torch.FloatTensor(mts[filename])
    # Predict
    x = {"US": x_us.unsqueeze(0).to(device),
        "DP": x_dp.unsqueeze(0).to(device),
        "TX": x_tx,
        "MT": x_mt.unsqueeze(0).to(device)}

    model = model.to(device)
    outputs = model(x).cpu().detach().numpy().round(4)
    # Save Outputs
    outputs_df = pd.DataFrame(outputs, columns=[f"Class_{i}" for i in range(outputs.shape[1])])
    outputs_df.to_csv("Outputs/prediction.csv", index=False)