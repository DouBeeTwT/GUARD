import torch
import torch.nn as nn
import torch.nn.functional as F

class ClipLoss(nn.Module):
    def __init__(self, temperature=0.07):
        super(ClipLoss, self).__init__()
        self.temperature = temperature

    def forward(self, image_embeddings, text_embeddings):
        logits_per_image = self._get_logits(image_embeddings, text_embeddings)
        logits_per_text = logits_per_image.T
        loss_i = torch.tensor(0.0, device=image_embeddings.device)
        loss_t = torch.tensor(0.0, device=image_embeddings.device)
        for i in range(image_embeddings.shape[0]):
            loss_i -= (
                torch.log(
                    torch.exp(logits_per_image[i, i])
                    / torch.exp(logits_per_image[i, :]).sum()
                )
                / image_embeddings.shape[0]
            )
            loss_t -= (
                torch.log(
                    torch.exp(logits_per_text[i, i])
                    / torch.exp(logits_per_text[i, :]).sum()
                )
                / image_embeddings.shape[0]
            )
        return (loss_i + loss_t) / 2

    def _get_logits(self, a, b):
        a = F.normalize(a, p=2, dim=-1)
        b = F.normalize(b, p=2, dim=-1)
        return torch.matmul(a, b.T) / self.temperature


class MultiClassFocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2, reduction='mean'):
        super(MultiClassFocalLoss, self).__init__()
        self.alpha = torch.tensor(alpha)  # 形状为 [num_classes]
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        # 计算Softmax概率
        log_probs = F.log_softmax(inputs, dim=1)
        probs = torch.exp(log_probs)
        
        # 获取真实类别对应的概率
        batch_size = targets.size(0)
        targets = targets.view(-1, 1)
        pt = probs.gather(1, targets).view(-1)  # shape: [batch_size]
        
        # 计算Focal Loss
        log_pt = log_probs.gather(1, targets).view(-1)
        focal_term = (1 - pt) ** self.gamma
        
        # 应用类别权重
        if self.alpha is not None:
            self.alpha = self.alpha.to(targets.device)
            alpha = self.alpha.gather(0, targets.view(-1))  # 按target选择对应α
            loss = -alpha * focal_term * log_pt
        else:
            loss = -focal_term * log_pt
        
        # 聚合损失
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss