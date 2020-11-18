import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd
from torch.nn import Parameter
import math

class CircleLoss(nn.Module):
    def __init__(self, in_features, out_features, s=64, m=0.35):
        super(CircleLoss, self).__init__()
        self.weight = Parameter(torch.Tensor(out_features, in_features))
        self._s = s
        self._m = m
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))

    def __call__(self, bn_feat, targets):

        sim_mat = F.linear(F.normalize(bn_feat), F.normalize(self.weight))
        alpha_p = F.relu(-sim_mat.detach() + 1 + self._m)
        alpha_n = F.relu(sim_mat.detach() + self._m)
        delta_p = 1 - self._m
        delta_n = self._m

        s_p = self._s * alpha_p * (sim_mat - delta_p)
        s_n = self._s * alpha_n * (sim_mat - delta_n)

        one_hot = torch.zeros(sim_mat.size(), device=targets.device)
        one_hot.scatter_(1, targets.view(-1, 1).long(), 1)

        pred_class_logits = one_hot * s_p + (1.0 - one_hot) * s_n

        return pred_class_logits