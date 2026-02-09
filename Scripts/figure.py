from Scripts.model import MMM, US_Model, MT_Model, DP_Model
from Scripts.explainer import figure_explainer, text_explainer, FNN_explainer
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
import matplotlib.pyplot as plt
from transformers import BertTokenizer, BertModel
from tqdm import tqdm
import pandas as pd
import numpy as np
import cv2
import os
import torch
import warnings
warnings.filterwarnings("ignore")

def explain_figure(filename:str, num_classes:int, ckpt_path:str, device:torch.device):
    alpha_dict = {2: [0.5, 0.5], 3: [0.12, 0.18, 0.7]}
    key_words_list = ["striped", "heterogeneous", "few", "around", "abundant",
                    "Isoechoic", "cysticsolid", "mixed", "thick", "obvious"]

    text = pd.read_csv("Dataset_Class3/echo_feature_text.csv", index_col=0).loc[filename, "Text"]
    mts = pd.read_csv('Dataset_Class3/matrix289.csv')
    mts = mts.set_index('Figure Pathway').apply(lambda x: x.values.tolist(), axis=1).to_dict()
    trans_list = pd.read_csv("Dataset_Class3/trans_list.csv", index_col=False)
    rgb_img = cv2.imread(os.path.join("Dataset_Class3/doppler/",filename))
    rgb_img = cv2.resize(rgb_img, (224, 224)) / 255.0

    ## Load Multi-modal model
    model_multi = MMM(lr=1e-4, mode = "US+DP+TX+MT+AA+CL",
                num_classes=num_classes, alpha=alpha_dict[num_classes],
                gamma=2, delta=0.5, device=device).eval()
    model_multi.load_state_dict(
        torch.load(ckpt_path, map_location=device)["state_dict"]
        )

    ## Load US model
    model_us = US_Model().eval()
    model_us.features_us.load_state_dict(model_multi.encoder_us.state_dict())

    ## DP模态分支
    model_dp = DP_Model().eval()
    model_dp.features_dp.load_state_dict(model_multi.encoder_dp.state_dict())

    ## MT模态分支
    model_mt = MT_Model().eval()
    model_mt.features_mt.load_state_dict(model_multi.encoder_mt.state_dict())

    ## TX模态分支

    model_tx = BertModel.from_pretrained('bert-base-uncased').eval()
    model_tx.load_state_dict(model_multi.encoder_tx.model.state_dict())
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')


    visualization_matrix = text_explainer(text=text,
                                        model=model_tx,
                                        tokenizer=tokenizer,
                                        model_pth=model_multi.encoder_tx.model.state_dict(), 
                                        fc_pth=model_multi.encoder_tx.fc.state_dict(),
                                        max=512)
    node_dict = visualization_matrix.idxmax().to_dict()


    target_layers = [model_us.features_us[-2][-1]]
    input_tensor_tx = torch.FloatTensor(rgb_img.transpose(2,0,1)).unsqueeze(dim=0)

    input_tensor_mt = torch.FloatTensor(mts[filename]).unsqueeze(dim=0).requires_grad_(True)
    mt_attention_matrix = FNN_explainer(model_mt,input_tensor_mt,max=512)
    mt_attention_matrix = pd.DataFrame(mt_attention_matrix)

    for key_word in key_words_list:
        if key_word in text:
            # US Explanation
            with GradCAM(model=model_us, target_layers=target_layers) as cam:
                grayscale_cam = cam(input_tensor=input_tensor_tx, targets=[ClassifierOutputTarget(node_dict[key_word]%512)])
                attention_map = grayscale_cam[0, :]
            # MT Explanation
            mt_attention_list = mt_attention_matrix.iloc[node_dict[key_word]%512,:].to_numpy()
            index_top = np.argsort(-np.abs(mt_attention_list))[:1]
            mt_name = trans_list.iloc[index_top]["Name"].to_list()[0]

            fig = plt.figure(figsize=(4,4))
            ax = fig.add_subplot(111)
            ax.imshow(rgb_img)
            ax.imshow(attention_map, cmap="jet",alpha=0.3)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.text(4,204,"TX: "+key_word,color="white")
            ax.text(4,20,"MT: "+mt_name,color="white")
            save_filename = filename.replace("/","_").split(".")[0]+"_"+key_word+".png"
            plt.savefig(os.path.join("Outputs/",save_filename), bbox_inches='tight', pad_inches=0)
    print("Sample {:13s} has done!".format(filename))

if __name__ == "__main__":
    num_classes = 2
    filename = "N1-2/0023.png"
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    ckpt_path = 'Checkpoints/best-model-class2.ckpt'
    results = explain_figure(filename, num_classes, ckpt_path, device)