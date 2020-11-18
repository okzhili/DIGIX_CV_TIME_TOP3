import torch.nn as nn
def count_num_param(model):
    num_param = sum(p.numel() for p in model.parameters()) / 1e+06

    if isinstance(model, nn.DataParallel):
        model = model.module

    if hasattr(model, 'classifier') and isinstance(model.classifier, nn.Module):
        # we ignore the classifier because it is unused at test time
        num_param -= sum(p.numel() for p in model.classifier.parameters()) / 1e+06
    return num_param