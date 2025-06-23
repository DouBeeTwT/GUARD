import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from tqdm import trange
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

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
        _, outputs = self.model(**inputs, return_dict=False) # outputs.pooler_output.shape = (1, 768)
        outputs = self.fc(outputs)
        outputs_logits = torch.nn.functional.sigmoid(outputs)
        outputs_logits = outputs_logits.flatten(1)
        target = torch.zeros_like(outputs_logits)
        target[0,label] = 1
        loss = torch.nn.functional.cross_entropy(outputs_logits, target)
        loss.backward()
        
        # 计算权重
        pooled_grad = torch.mean(self.gradients, dim=[0, 1])
        activations = self.activations.squeeze(0)
        
        # 生成贡献度
        for i in range(activations.size(0)):
            activations[i] *= pooled_grad[i]
            
        weights = torch.mean(activations, dim=1)
        return weights.tolist()
    
def grand_cam_text(model, model_pth, fc_pth, inputs, tokenizer, N=1):
    explainer = BertGradCAM(model)
    explainer.model.load_state_dict(model_pth)
    explainer.fc.load_state_dict(fc_pth)
    visualization_matrix = pd.DataFrame()

    for l in trange(N, leave=False, ncols=60):
        contributions = explainer.explain(inputs, label=l)
        contributions = np.array(contributions)[1:-1]
        contributions = contributions - contributions.min()
        contributions = contributions / contributions.sum()

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
        
        # 把visualization_word_list添加到visualization_matrix中
        temp_df = pd.DataFrame.from_dict(visualization_word_list, orient='index').T
        visualization_matrix = pd.concat([visualization_matrix, temp_df], axis=0, ignore_index=True)

    return visualization_matrix

class FNNGradCAM:
    def __init__(self, model):
        self.model = model
        self.gradients = None
        self.activations = None
        
        # 注册钩子获取梯度
        layer = model.features_mt
        layer.register_forward_hook(self._get_activations)
        layer.register_backward_hook(self._get_gradients)

    def _get_activations(self, module, input, output):
        self.activations = input[0]

    def _get_gradients(self, module, grad_input, grad_output):
        # 获取输入特征的梯度
        self.gradients = grad_output[0]  # 使用grad_input而不是grad_output

    def explain(self, inputs, label):
        outputs = self.model(inputs)
        outputs_logits = torch.nn.functional.sigmoid(outputs)
        outputs_logits = outputs_logits.flatten(1)
        target = torch.zeros_like(outputs_logits)
        target[0,label] = 1
        loss = torch.nn.functional.cross_entropy(outputs_logits, target)
        loss.backward()

         # 计算权重
        grid_list=[]
        for name, param in model.named_parameters():
            if 'weight' in name:
                grid_list.append(param.grad)
        grid_matrix = torch.matmul(torch.matmul(grid_list[2], grid_list[1],), grid_list[0],)
        #grid_matrix = torch.matmul(self.activations,grid_matrix)
        pooled_grad = grid_matrix[label]
        #pooled_grad = pooled_grad*self.activations[0]

        return pooled_grad.tolist()

def calc(model, input, label):
    output = model(input)
    target = torch.zeros_like(output)
    target[0,label] = 1
    model.zero_grad()
    output.backward(gradient=target, retain_graph=True)
    input_grad = input.grad.data
    contribution = input_grad.squeeze()

    return contribution.tolist()