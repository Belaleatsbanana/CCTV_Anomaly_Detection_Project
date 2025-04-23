import torch
from torch.utils.data import Dataset
import numpy as np
import os
import random
from pathlib import Path
from paths import PathConfig  # Import your PathConfig class

class Normal_Loader(Dataset):
    """
    is_train = 1 <- train, 0 <- test
    """
    def __init__(self, is_train=1):
        super(Normal_Loader, self).__init__()
        self.is_train = is_train
        self.path_config = PathConfig()  # Initialize path configuration
        
        # Get appropriate file list
        data_file = 'train_normal.txt' if self.is_train else 'test_normalv2.txt'
        data_path = self.path_config.processed_data / data_file
        
        with open(data_path, 'r') as f:
            self.data_list = [line.strip() for line in f.readlines()]
            
        if not self.is_train:
            random.shuffle(self.data_list)
            self.data_list = self.data_list[:-10]

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        if self.is_train:
            base_name = self.data_list[idx]
            rgb_path = self.path_config.rgb_features / f"{base_name}.npy"
            flow_path = self.path_config.flow_features / f"{base_name}.npy"
            return np.concatenate([np.load(rgb_path), np.load(flow_path)], axis=1)
            
        else:
            parts = self.data_list[idx].split(' ')
            name, frames, gts = parts[0], int(parts[1]), int(parts[2])
            
            rgb_path = self.path_config.rgb_features / f"{name}.npy"
            flow_path = self.path_config.flow_features / f"{name}.npy"
            features = np.concatenate([np.load(rgb_path), np.load(flow_path)], axis=1)
            
            return features, gts, frames

class Anomaly_Loader(Dataset):
    """
    is_train = 1 <- train, 0 <- test
    """
    def __init__(self, is_train=1):
        super(Anomaly_Loader, self).__init__()
        self.is_train = is_train
        self.path_config = PathConfig()  # Initialize path configuration
        
        # Get appropriate file list
        data_file = 'train_anomaly.txt' if self.is_train else 'test_anomalyv2.txt'
        data_path = self.path_config.processed_data / data_file
        
        with open(data_path, 'r') as f:
            self.data_list = [line.strip() for line in f.readlines()]

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        if self.is_train:
            base_name = self.data_list[idx]
            rgb_path = self.path_config.rgb_features / f"{base_name}.npy"
            flow_path = self.path_config.flow_features / f"{base_name}.npy"
            return np.concatenate([np.load(rgb_path), np.load(flow_path)], axis=1)
            
        else:
            parts = self.data_list[idx].split('|')
            name = parts[0].strip()
            frames = int(parts[1].strip())
            gts = [int(x.strip()) for x in parts[2].strip('[]').split(',')]
            
            rgb_path = self.path_config.rgb_features / f"{name}.npy"
            flow_path = self.path_config.flow_features / f"{name}.npy"
            features = np.concatenate([np.load(rgb_path), np.load(flow_path)], axis=1)
            
            return features, gts, frames

if __name__ == '__main__':
    # Test the dataset configuration
    config = PathConfig()
    print("Testing dataset paths:")
    print(f"Base path: {config.base}")
    print(f"RGB features path: {config.rgb_features.exists()}")
    print(f"Flow features path: {config.flow_features.exists()}")
    
    normal_train = Normal_Loader(is_train=1)
    print(f"\nNormal training samples: {len(normal_train)}")
    sample = normal_train[0]
    print(f"Sample shape: {sample.shape}")