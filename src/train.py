from torch.utils.data import DataLoader
from dataset import Normal_Loader, Anomaly_Loader
from models import Learner
from loss import MIL
import numpy as np
from sklearn import metrics
import torch

# Initialize datasets
normal_train = Normal_Loader(is_train=1)
normal_test = Normal_Loader(is_train=0)
anomaly_train = Anomaly_Loader(is_train=1)
anomaly_test = Anomaly_Loader(is_train=0)

# Create dataloaders
batch_size = 30
normal_train_loader = DataLoader(normal_train, batch_size=batch_size, shuffle=True)
anomaly_train_loader = DataLoader(anomaly_train, batch_size=batch_size, shuffle=True)
normal_test_loader = DataLoader(normal_test, batch_size=1, shuffle=False)
anomaly_test_loader = DataLoader(anomaly_test, batch_size=1, shuffle=False)

# Model setup
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = Learner().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=5)

def train(epoch):
    model.train()
    total_loss = 0.0
    
    for (normal, anomaly) in zip(normal_train_loader, anomaly_train_loader):
        inputs = torch.cat([anomaly, normal], dim=1).view(-1, 2048).to(device)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = MIL(outputs, batch_size)
        
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    print(f'Epoch {epoch} - Train Loss: {total_loss/len(normal_train_loader):.4f}')
    return total_loss/len(normal_train_loader)

def test():
    model.eval()
    all_scores = []
    all_labels = []
    
    with torch.no_grad():
        # Process anomaly videos
        for data in anomaly_test_loader:
            features, gts, frames = data
            features = features.view(-1, 2048).to(device)
            scores = model(features).cpu().numpy().flatten()
            
            num_frames = int(frames.item())  # Convert to integer
            frame_scores = np.zeros(num_frames)
            step = np.linspace(0, num_frames//16, 33, dtype=int)
            
            # Create frame-level scores
            for j in range(32):
                start = step[j] * 16
                end = min(step[j+1] * 16, num_frames)
                frame_scores[start:end] = scores[j]
            
            # Create binary labels
            gt_array = np.zeros(num_frames, dtype=int)
            for k in range(0, len(gts), 2):
                start_frame = gts[k] - 1  # Convert to 0-based index
                end_frame = min(gts[k+1], num_frames)
                gt_array[start_frame:end_frame] = 1
            
            all_scores.extend(frame_scores.tolist())
            all_labels.extend(gt_array.tolist())
        
        # Process normal videos
        for data in normal_test_loader:
            features, _, frames = data
            features = features.view(-1, 2048).to(device)
            scores = model(features).cpu().numpy().flatten()
            
            num_frames = int(frames.item())
            frame_scores = np.zeros(num_frames)
            step = np.linspace(0, num_frames//16, 33, dtype=int)
            
            for j in range(32):
                start = step[j] * 16
                end = min(step[j+1] * 16, num_frames)
                frame_scores[start:end] = scores[j]
            
            all_scores.extend(frame_scores.tolist())
            all_labels.extend(np.zeros(num_frames, dtype=int).tolist())
    
    # Verify matching lengths
    assert len(all_scores) == len(all_labels), \
        f"Length mismatch: scores {len(all_scores)}, labels {len(all_labels)}"
    
    # Calculate AUC
    fpr, tpr, _ = metrics.roc_curve(all_labels, all_scores)
    auc = metrics.auc(fpr, tpr)
    print(f'Test AUC: {auc:.4f}')
    return auc

# Training loop
best_auc = 0
for epoch in range(75):
    train_loss = train(epoch)
    current_auc = test()
    
    scheduler.step(current_auc)
    
    if current_auc > best_auc:
        best_auc = current_auc
        torch.save(model.state_dict(), f'best_model_auc{current_auc:.4f}.pth')

print(f'Best AUC achieved: {best_auc:.4f}')