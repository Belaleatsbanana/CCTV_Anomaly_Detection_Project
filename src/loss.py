import torch
import torch.nn.functional as F

def MIL(y_pred, batch_size):
    loss = torch.tensor(0.).cuda()
    sparsity = torch.tensor(0.).cuda()
    smooth = torch.tensor(0.).cuda()
    
    y_pred = y_pred.view(batch_size, -1)
    
    for i in range(batch_size):
        # Corrected permutation indices
        anomaly_idx = torch.randperm(32).cuda()
        normal_idx = torch.randperm(32).cuda()
        
        y_anomaly = y_pred[i, :32][anomaly_idx]
        y_normal = y_pred[i, 32:][normal_idx]
        
        # Loss components
        loss += F.relu(1 - y_anomaly.max() + y_normal.max())
        sparsity += y_anomaly.sum() * 8e-5
        smooth += (y_pred[i, :31] - y_pred[i, 1:32]).pow(2).sum() * 8e-5
    
    return (loss + sparsity + smooth) / batch_size