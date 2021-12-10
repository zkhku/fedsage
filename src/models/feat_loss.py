import numpy as np
import torch
import torch.nn.functional as F
from src.utils import config


def greedy_loss(pred_feats, true_feats,pred_missing,true_missing):
    if config.cuda:
        true_missing=true_missing.cpu()
        pred_missing = pred_missing.cpu()
    loss=torch.zeros(pred_feats.shape)
    if config.cuda:
        loss=loss.cuda()
    pred_len=len(pred_feats)
    pred_missing_np = pred_missing.detach().numpy().reshape(-1).astype(np.int32)
    true_missing_np = true_missing.detach().numpy().reshape(-1).astype(np.int32)
    true_missing_np = np.clip(true_missing_np,0,config.num_pred)
    pred_missing_np = np.clip(pred_missing_np, 0, config.num_pred)
    for i in range(pred_len):
        for pred_j in range(min(config.num_pred, pred_missing_np[i])):
            if true_missing_np[i]>0:
                if isinstance(true_feats[i][true_missing_np[i]-1], np.ndarray):
                    true_feats_tensor = torch.tensor(true_feats[i][true_missing_np[i]-1])
                    if config.cuda:
                        true_feats_tensor=true_feats_tensor.cuda()
                else:
                    true_feats_tensor=true_feats[i][true_missing_np[i]-1]
                loss[i][pred_j] += F.mse_loss(pred_feats[i][pred_j].unsqueeze(0).float(),
                                                  true_feats_tensor.unsqueeze(0).float()).squeeze(0)

                for true_k in range(min(config.num_pred, true_missing_np[i])):
                    if isinstance(true_feats[i][true_k], np.ndarray):
                        true_feats_tensor = torch.tensor(true_feats[i][true_k])
                        if config.cuda:
                            true_feats_tensor = true_feats_tensor.cuda()
                    else:
                        true_feats_tensor = true_feats[i][true_k]

                    loss_ijk=F.mse_loss(pred_feats[i][pred_j].unsqueeze(0).float(),
                                        true_feats_tensor.unsqueeze(0).float()).squeeze(0)
                    if torch.sum(loss_ijk)<torch.sum(loss[i][pred_j].data):
                        loss[i][pred_j]=loss_ijk
            else:
                continue
    return loss
