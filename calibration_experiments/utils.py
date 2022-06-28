import numpy as np
import torch

def evaluate(lowers, uppers, labels, confidence_intervals):

        containment = ((lowers <= labels) * (labels <= uppers)).float()
        percent_contain = containment.mean(dim=0) * 100

        return containment, percent_contain - torch.tensor(confidence_intervals, device=labels.device)

def cumsum(tensor):
    output = torch.zeros_like(tensor)

    for i in range(tensor.shape[1]):
        output[:, i] = tensor[:, i] + output[:, max(0, i-1)].detach()
    
    return output

def detach_tensor(tensor):
    if tensor.device != 'cpu':
        return tensor.detach().cpu()
    return tensor

def round_tensor(tensor, deci):
    return np.around(detach_tensor(tensor).numpy(), deci)

def get_sci_exp(tensor):
    assert (tensor < 0).sum() == 0 # doesn't handle negative
    return torch.floor(torch.log10(tensor))

def rms(containments, confidence_intervals):
    containments = detach_tensor(containments)
    return torch.sqrt(((containments - torch.tensor(confidence_intervals)) ** 2).mean())

def adaptive_binning_rms(containments, labels, confidence_intervals, bin_size=100):
    if not torch.is_tensor(containments):
        containments = torch.cat(containments)
    if not torch.is_tensor(labels):
        labels = torch.cat(labels)
    containments = detach_tensor(containments[labels.argsort()])
    squared_error = []
    N, _ = containments.shape
    for i in range(0, N, bin_size):
        squared_error.append((containments[i:i+bin_size].mean(dim=0) * 100 - torch.tensor(confidence_intervals)) ** 2)
    
    squared_error = torch.stack(squared_error)

    return torch.sqrt(squared_error.mean(dim=0))

