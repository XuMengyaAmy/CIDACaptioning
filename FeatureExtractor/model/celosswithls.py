'''
Project : Incremental Domain Adaptation with Smoothing and Calibration for Surgical Report Generation
Note    : Lable smoothing loss
'''
import torch
import torch.nn as nn
import torch.nn.functional as F

class CELossWithLS(torch.nn.Module):
    def __init__(self, smoothing=0.1, gamma=3.0, isCos=True, ignore_index=-1):
        super(CELossWithLS, self).__init__()
        self.complement = 1.0 - smoothing
        self.smoothing = smoothing
        self.log_softmax = nn.LogSoftmax(dim=1)
        self.gamma = gamma
        self.ignore_index = ignore_index

    def forward(self, logits, target):
        with torch.no_grad():
            smoothen_ohlabel = target * self.complement + self.smoothing / target.shape[1]
        
        target_labels = torch.argmax(target, dim=1)
        #print(target_labels)
        logs = self.log_softmax(logits[target_labels!=self.ignore_index])
        pt = torch.exp(logs)
        return -torch.sum((1-pt).pow(self.gamma)*logs * smoothen_ohlabel[target_labels!=self.ignore_index], dim=1).mean()



