from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
import cv2
import matplotlib.pyplot as plt
from tqdm import trange
import numpy as np
import torch
import torch.nn as nn
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

def figure_explainer(figure_path:str, model, target_layers, N=None, max=None, show=False):
    rgb_img = cv2.imread(figure_path)
    rgb_img = cv2.resize(rgb_img, (224, 224)) / 255.0
    input_tensor = torch.FloatTensor(rgb_img.transpose(2,0,1)).unsqueeze(dim=0)

    grayscale_cam_list = []
    with GradCAM(model=model, target_layers=target_layers) as cam:
        if max is not None:
            for i in trange(max):
                grayscale_cam = cam(input_tensor=input_tensor, targets=[ClassifierOutputTarget(i)])
                grayscale_cam = grayscale_cam[0, :]
                grayscale_cam_list.append(grayscale_cam)
        elif N is not None:
            grayscale_cam = cam(input_tensor=input_tensor, targets=[ClassifierOutputTarget(N)])
            grayscale_cam = grayscale_cam[0, :]
            grayscale_cam_list.append(grayscale_cam)
        else:
            print("ERROR: N or max must be specified. N for sigle image, max for all images.")
    grayscale_cam_list = np.array(grayscale_cam_list)
    
    if show:
        if grayscale_cam_list.shape[0] > 1:
            grayscale_cam_max = np.argmax(grayscale_cam_list, axis=0)
        else:
            grayscale_cam_max = grayscale_cam_list[0]
        plt.imshow(rgb_img)
        plt.imshow(grayscale_cam_max, cmap="Reds",alpha=0.2)
        plt.xticks([])
        plt.yticks([])
        plt.show()
    return grayscale_cam_list

class BertGradCAM:
    def __init__(self, model):
        self.model = model
        self.fc = nn.Linear(self.model.config.hidden_size, 512)
        self.gradients = None
        self.activations = None
        
        # 注册钩子获取梯度
        layer = model.encoder.layer[-1]
        layer.register_forward_hook(self._get_activations)
        layer.register_backward_hook(self._get_gradients)

    def _get_activations(self, module, input, output):
        self.activations = output[0]

    def _get_gradients(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]

    def explain(self, inputs, label):
        _, outputs = self.model(**inputs, return_dict=False)
        outputs = self.fc(outputs)
        outputs_logits = torch.nn.functional.sigmoid(outputs)
        outputs_logits = outputs_logits.flatten(1)
        target = torch.zeros_like(outputs_logits)
        target[0,label] = 1
        loss = torch.nn.functional.cross_entropy(outputs_logits, target)
        loss.backward()

        pooled_grad = torch.mean(self.gradients, dim=[0, 1])
        activations = self.activations.squeeze(0)
        
        for i in range(activations.size(0)):
            activations[i] *= pooled_grad[i]
            
        weights = torch.mean(activations, dim=1)
        return weights.tolist()

def combine_tokens(inputs, tokenizer, contributions):
    tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])[1:-1]
    visualization = list(zip(tokens, contributions))

    visualization_word_list = {}
    key = ""
    attention = 0.0
    for word, value in visualization:
        if word[0] == "#":
            key += word.split("#")[-1]
            attention += value
        elif word[0] != ",":
            if key != "":
                if key not in visualization_word_list:
                    visualization_word_list[key] = []  # 初始化新键
                visualization_word_list[key].append(round(attention,4))
            key = word
            attention = value
    return visualization_word_list

def text_explainer(text, model, tokenizer, model_pth, fc_pth, N=None, max=None):
    explainer = BertGradCAM(model)
    explainer.model.load_state_dict(model_pth)
    explainer.fc.load_state_dict(fc_pth)
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
    visualization_matrix = pd.DataFrame()

    if max is not None:
        for l in trange(max, ncols=80):
            contributions = explainer.explain(inputs, label=l)
            contributions = np.array(contributions)[1:-1]
            contributions = contributions - contributions.min()
            contributions = contributions / contributions.sum()
            visualization_word_list = combine_tokens(inputs, tokenizer, contributions)
            temp_df = pd.DataFrame.from_dict(visualization_word_list, orient='index').T
            visualization_matrix = pd.concat([visualization_matrix, temp_df], axis=0, ignore_index=True)
    else:
        contributions = explainer.explain(inputs, label=N)
        contributions = np.array(contributions)[1:-1]
        contributions = contributions - contributions.min()
        contributions = contributions / contributions.sum()
        visualization_word_list = combine_tokens(inputs, tokenizer, contributions)
        temp_df = pd.DataFrame.from_dict(visualization_word_list, orient='index').T
        visualization_matrix = pd.concat([visualization_matrix, temp_df], axis=0, ignore_index=True)


    return visualization_matrix

def contribution_calc(model, input, label):
    output = model(input)
    target = torch.zeros_like(output)
    target[0,label] = 1
    model.zero_grad()
    output.backward(gradient=target, retain_graph=True)
    input_grad = input.grad.data
    contribution = input_grad.squeeze()
    return contribution.tolist()

def FNN_explainer(model, input, N=None, max=None):
    if max is not None:
        mt_attention_matrix = [contribution_calc(model,input,i) for i in trange(max, ncols=80)]
    elif N is not None:
        mt_attention_matrix = [contribution_calc(model,input,N)]
    return np.array(mt_attention_matrix)