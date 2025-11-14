from typing import Optional

import torch
from torch import nn
from .utils import classify_scenarios

def SAD_loss(y_hat: torch.Tensor, dur: float):
    return 10*torch.log10(torch.sum(y_hat**2, dim = -1)/(dur+1e-6) + 1e-6)

def SAD_losses(y_hat: torch.Tensor, dur: float, weights: Optional[list] = None):
    loss = 0
    for yh, w in zip(y_hat, weights):
        loss += SAD_loss(yh, dur)*w
    return loss

def SI_SNR_loss(y_hat: torch.Tensor, 
                y: torch.Tensor):
    reference_energy = torch.sum(y**2, dim = -1, keepdim = True) + 1e-6
    optimal_scaling = (y_hat*y).sum(dim = -1, keepdim = True)/reference_energy
    projection = optimal_scaling*y
    noise = y_hat - projection
    ratio = torch.sum(projection**2, dim = -1)/(torch.sum(noise**2, dim = -1) + 1e-6)
    return -10*torch.log10(ratio + 1e-6)

def SI_SNR_losses(y_hat: torch.Tensor, y: torch.Tensor, weights: Optional[list] = None):
    loss = 0
    for yh, w in zip(y_hat, weights):
        loss += SI_SNR_loss(yh, y)*w
    return loss
    
def DIA_loss(y_hat: torch.Tensor, y: torch.Tensor):
    """
    y_hat, y: Tensor. (N, k+1, T)
    """
    return nn.functional.binary_cross_entropy(y_hat, y, reduction = 'mean')

def DIA_losses(y_hat: torch.Tensor, y: list, subsampling: int):
    loss = 0
    y = torch.stack(y)[:, :, ::subsampling]
    for yh in y_hat:
        loss += DIA_loss(yh.transpose(1, 2), y)
    return loss

def SPK_losses(y_hat: torch.Tensor, y: torch.Tensor):
    """
    y_hat: Tensor, (N, Dspk)
    y: Tensor, (N)
    """
    loss = 0
    count = 0
    for pred, gt in zip(y_hat, y):
        pred = pred[gt != -1]
        gt = gt[gt != -1]
        loss += nn.functional.cross_entropy(pred, gt, reduction = 'sum')
        count += len(pred)
    return loss/count

def EXT_losses(ext_preds: list, src_tars: list, masks: list, 
               glob_weights: list = [1, 1, 1, 1], 
               local_weights: list = [1, 1, 1], 
               sr: int = 16000):
    total_loss = 0
    for exts, srcs, msks in zip(ext_preds, src_tars, masks):
        local_loss = 0
        for i, (ext, src) in enumerate(zip(exts, srcs)):
            QQ, QS, SQ, SS = classify_scenarios(msks, target_dim = i, alpha = 3)
            
            if torch.sum(src[src > 0]) == 0:
                QS = QS + SQ + SS
                L_EQQ = SAD_losses(ext*QQ.unsqueeze(0), torch.sum(QQ)/sr, weights = local_weights)
                L_EQS = SAD_losses(ext*QS.unsqueeze(0), torch.sum(QS)/sr, weights = local_weights)
                local_loss += glob_weights[0]*L_EQQ + glob_weights[1]*L_EQS
                continue
            
            if torch.sum(QQ) != 0:
                L_EQQ = SAD_losses(ext*QQ.unsqueeze(0), torch.sum(QQ)/sr, weights = local_weights)
            else:
                L_EQQ = 0
            
            if torch.sum(QS) != 0:
                L_EQS = SAD_losses(ext*QS.unsqueeze(0), torch.sum(QS)/sr, weights = local_weights)
            else:
                L_EQS = 0
            
            if torch.sum(SS) != 0:
                L_SSS = SI_SNR_losses(ext*SS.unsqueeze(0), src.squeeze(0)*SS, weights = local_weights)
            else:
                L_SSS = 0
            
            if torch.sum(SQ) != 0:
                L_SSQ = SI_SNR_losses(ext*SQ.unsqueeze(0), src.squeeze(0)*SQ, weights = local_weights)
            else:
                L_SSQ = 0
            local_loss += glob_weights[0]*L_EQQ + glob_weights[1]*L_EQS + glob_weights[2]*L_SSS + glob_weights[3]*L_SSQ
        total_loss += local_loss/len(exts)
    return total_loss/len(ext_preds)