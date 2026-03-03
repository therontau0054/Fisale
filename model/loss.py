import torch

def relative_l2_loss(x, y, is_train = False):
    b = x.shape[0]
    if is_train:
        diff_norms = torch.norm(x.reshape(b, -1) - y.reshape(b, -1), p = 2, dim = -1)
        y_norms = torch.norm(y.reshape(b, -1), p = 2, dim = -1)
        return torch.mean(diff_norms / y_norms)
    else:
        ## calculate relative error for each channel and then average 
        ## to avoid the influence of different channel scales and units
        error_abs = torch.sum(torch.abs(x - y), dim = -2)
        y_abs = torch.sum(torch.abs(y), dim = -2)
        return (error_abs / y_abs).mean()

def mse_l2_loss(x, y):
    return  torch.nn.MSELoss()(x, y)