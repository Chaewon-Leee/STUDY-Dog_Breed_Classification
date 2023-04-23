import torch
import torch.nn as nn

class LabelSmoothingCrossEntropy(nn.Module):
    def __init__(self, classes=120, smoothing=0.1, dim=-1):
        super(LabelSmoothingCrossEntropy, self).__init__()
        self.confidence = 1.0 - smoothing # 0.9
        self.smoothing = smoothing # 0.1
        self.classes = classes
        self.dim = dim

    def forward(self, pred, target):
        true_probs = torch.zeros_like(pred)
        true_probs.fill_(self.smoothing / (self.classes - 1))
        # 모든 클래스에 대해 smoothing / (classes - 1)의 확률을 할당
        true_probs.scatter_(1, target.data.unsqueeze(1), self.confidence)
        # 정답 인덱스의 정답 확률을 confidence로 설정
        return torch.mean(torch.sum(true_probs * -pred, dim=self.dim)) # negative log likelihood loss
