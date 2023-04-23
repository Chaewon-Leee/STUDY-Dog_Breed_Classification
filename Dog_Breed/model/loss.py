import torch.nn.functional as F
from label_smoothing import LabelSmoothingCrossEntropy

def nll_loss(output, target):
    return F.nll_loss(output, target)

def label_smoothing_loss(output, target):
    return LabelSmoothingCrossEntropy()(output, target)
