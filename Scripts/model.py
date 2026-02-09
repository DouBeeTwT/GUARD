import math
import torch
import torchmetrics as tm
from torch import optim
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import CosineAnnealingLR
import pytorch_lightning as pl
from torchvision.models import resnet18, ResNet18_Weights, resnet50, ResNet50_Weights
from .loss import ClipLoss, MultiClassFocalLoss
import warnings

warnings.filterwarnings("ignore")


class TextTransformerEncoder(nn.Module):
    def __init__(self, model_name="BERT", output_dim=512, device="cuda"):
        super(TextTransformerEncoder, self).__init__()
        self.device = device
        if model_name == "BERT":
            from transformers import BertModel, BertTokenizer

            self.model = BertModel.from_pretrained("bert-base-uncased")
            self.fc = nn.Linear(self.model.config.hidden_size, output_dim)
            self.relu = nn.ReLU()
            self.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        elif model_name == "CLIP":
            from transformers import CLIPTextModel, CLIPTokenizer

            self.model = CLIPTextModel.from_pretrained("openai/clip-vit-base-patch16")
            self.fc = nn.Linear(self.model.config.hidden_size, output_dim)
            self.tokenizer = CLIPTokenizer.from_pretrained(
                "openai/clip-vit-base-patch16"
            )

    def forward(self, text):
        inputs = self.tokenizer(
            text, return_tensors="pt", padding=True, truncation=True
        )
        inputs = {key: value.to(self.device) for key, value in inputs.items()}
        self.model = self.model.to(self.device)
        self.fc = self.fc.to(self.device)
        _, pooled_output = self.model(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            return_dict=False,
        )
        compressed_features = self.relu(self.fc(pooled_output))
        return compressed_features

class MMM(pl.LightningModule):
    def __init__(self,
                 lr = 1e-5,
                 single_decoder = False,
                 mode = "US+DP+TX+MT+CL",
                 num_classes = 2,
                 alpha = None,
                 gamma = 2,
                 delta = 0.5,
                 device = "cuda"):
        super(MMM, self).__init__()

        self.lr = lr
        self.single_decoder = single_decoder
        self.mode = mode
        self.num_classes = num_classes
        self.alpha = alpha
        self.gamma = gamma
        self.delta = delta
        heads = 0

        if "US" in self.mode:
            heads += 1
            self.encoder_us = nn.Sequential(
                *list(resnet18(weights=ResNet18_Weights.DEFAULT).children())[:-1]
            )
            self.decoder_us = nn.Sequential(
                nn.Linear(in_features=512,out_features=num_classes),nn.Sigmoid()
            )
        if "DP" in self.mode:
            heads += 1
            self.encoder_dp = nn.Sequential(
                *list(resnet18(weights=ResNet18_Weights.DEFAULT).children())[:-1]
            )
            self.decoder_dp = nn.Sequential(
                nn.Linear(in_features=512,out_features=num_classes),nn.Sigmoid()
            )
        if "TX" in self.mode:
            heads += 1
            self.encoder_tx = TextTransformerEncoder(output_dim=512, device=device)
            self.decoder_tx = nn.Sequential(
                nn.Linear(in_features=512,out_features=num_classes),nn.Sigmoid()
            )
        if "MT" in self.mode:
            heads += 1
            self.encoder_mt = nn.Sequential(
                nn.Linear(in_features=289, out_features=1024),
                nn.Dropout(0.1),
                nn.ReLU(),
                nn.Linear(in_features=1024, out_features=512),
                nn.Dropout(0.1),
                nn.ReLU(),
                nn.Linear(in_features=512, out_features=512),
            )
            self.decoder_mt = nn.Sequential(
                nn.Linear(in_features=512,out_features=num_classes),nn.Sigmoid()
            )
        self.attention = nn.Sequential(nn.Linear(in_features=512*heads, out_features=512*heads))
        self.decoder = nn.Sequential(nn.Linear(in_features=512*heads, out_features=512),
                                        nn.Dropout(0.1),
                                        nn.ReLU(),
                                        nn.Linear(in_features=512, out_features=num_classes),
                                        nn.Softmax())
        
        self.loss1 = MultiClassFocalLoss(alpha=self.alpha, gamma=self.gamma)
        self.loss2 = ClipLoss()
        
        self.acc = tm.Accuracy(task="multiclass", num_classes= self.num_classes)
        self.auc = tm.AUROC(task="multiclass", num_classes= self.num_classes)
        self.sen = tm.Recall(task="multiclass", num_classes= self.num_classes)
        self.spe = tm.Specificity(task="multiclass", num_classes= self.num_classes)

    def entropy(self, preds):
        probs = F.softmax(preds, dim=1)
        entropy = -torch.sum(probs * torch.log(probs + 1e-8), dim=1)
        return entropy

    def get_features(self, x, mode=None):
        preds = None
        if mode == "US":
            features = self.encoder_us(x)
            features = torch.flatten(features, 1)
            if "AA" in self.mode:
                preds = self.decoder_us(features)
                w = self.entropy(preds)
                features = features * w.unsqueeze(1).expand(-1, 512)
        elif mode == "DP":
            features = self.encoder_dp(x)
            features = torch.flatten(features, 1)
            if "AA" in self.mode:
                preds = self.decoder_dp(features)
                w = self.entropy(preds)
                features = features * w.unsqueeze(1).expand(-1, 512)
        elif mode == "TX":
            features = self.encoder_tx(x)
            if "AA" in self.mode:
                preds = self.decoder_tx(features)
                w = self.entropy(preds)
                features = features * w.unsqueeze(1).expand(-1, 512)
        elif mode == "MT":
            features = self.encoder_mt(x) / 100.0
            if "AA" in self.mode:
                preds = self.decoder_mt(features)
                w = self.entropy(preds)
                features = features * w.unsqueeze(1).expand(-1, 512)
        return features, preds

    def forward(self, x, return_features=False):
        features = {}
        preds = {}
        for Name in ["US", "DP", "TX", "MT"]:
            if Name in self.mode:
                features[Name], preds[Name] = self.get_features(x[Name], mode=Name)
        outputs = torch.cat([v for v in features.values()], dim=1)
        if "AA" in self.mode:
            attention = self.attention(outputs)
            outputs = outputs*attention
        outputs = self.decoder(outputs)

        if return_features:
            return outputs, features, preds
        return outputs

    def common_step(self, batch, batch_idx):
        us, dp, tx, mt, labels = batch
        x = {"US": us, "DP": dp, "TX": tx, "MT": mt}
        outputs, features, predictions = self.forward(x, return_features=True)
        _, preds = outputs.max(1)
        return outputs, preds, features, predictions, labels
    
    def loss_step(self, outputs, features, predictions, labels):
        loss_class = torch.tensor(0.0, device=labels.device)
        loss_class += self.loss1(outputs, labels)
        loss_cross = torch.tensor(0.0, device=labels.device)
        if "CL" in self.mode:
            keys = list(features.keys())
            for i in range(len(keys)):
                for j in range(i + 1, len(keys)):
                    loss_cross += self.loss2(features[keys[i]], features[keys[j]])
            loss_cross /= max((len(keys) - 1) * (len(keys) - 2),1)
            loss = self.delta * loss_class + (1 - self.delta) * loss_cross
        else:
            loss = loss_class
        if "AA" in self.mode:
            loss += self.loss1(predictions["US"], labels)
            loss += self.loss1(predictions["DP"], labels)
            loss += self.loss1(predictions["TX"], labels)
            loss += self.loss1(predictions["MT"], labels)
        return loss
    
    def training_step(self, batch, batch_idx):
        outputs, preds, features, predictions, labels = self.common_step(batch, batch_idx)
        loss = self.loss_step(outputs, features, predictions, labels)
        
        acc = self.acc(preds, labels)
        self.log("Train/ACC", acc, prog_bar=True, on_step=False,on_epoch=True)
        self.log("Train/loss",loss,prog_bar=False, on_step=False,on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        outputs, preds, features, predictions, labels = self.common_step(batch, batch_idx)
        loss = self.loss_step(outputs, features, predictions, labels)

        acc = self.acc(preds, labels)
        auc = self.auc(outputs, labels)
        sen = self.sen(preds, labels)
        spe = self.spe(preds, labels)
        self.log_dict({"Val/loss": loss,
                       "Val/ACC": acc,
                       "Val/AUC": auc,
                       "Val/SEN": sen,
                       "Val/SPE": spe},
                       prog_bar=True, on_step=False, on_epoch=True)
        return outputs

    def prediction_step(self, batch, batch_idx):
        outputs, preds, features, predictions, labels = self.common_step(batch, batch_idx)
        return outputs, preds, labels
        
    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.lr, weight_decay=0.01)
        scheduler = CosineAnnealingLR(optimizer, T_max=100, eta_min=1e-6)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",
            }
        }

class US_Model(nn.Module):
    def __init__(self):
        super(US_Model, self).__init__()
        self.features_us = nn.Sequential(*list(resnet18(weights=ResNet18_Weights.DEFAULT).children())[:-1])
    def forward(self, x):
        features = self.features_us(x)
        features = torch.flatten(features, 1)
        return features

class MT_Model(nn.Module):
    def __init__(self):
        super(MT_Model, self).__init__()
        self.features_mt = nn.Sequential(
            nn.Linear(in_features=289, out_features=1024),
            nn.Dropout(0.1),
            nn.ReLU(),
            nn.Linear(in_features=1024, out_features=512),
            nn.Dropout(0.1),
            nn.ReLU(),
            nn.Linear(in_features=512, out_features=512)
        )
    def forward(self, x):
        features = self.features_mt(x) / 100.0
        features = torch.flatten(features, 1)
        return features
    
class DP_Model(nn.Module):
    def __init__(self, original_model=None, device="cuda"):
        super().__init__()
        self.features_dp = nn.Sequential(*list(resnet18(weights=ResNet18_Weights.DEFAULT).children())[:-1])
    def forward(self, x):
        features = self.features_dp(x)
        features = torch.flatten(features, 1)
        return features