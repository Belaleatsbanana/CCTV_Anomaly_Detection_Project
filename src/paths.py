from pathlib import Path
import os

class PathConfig:
    def __init__(self):
        self.on_colab = 'COLAB_GPU' in os.environ
        self.base = Path('/content/drive/MyDrive/ANOMALY_DETECTION') if self.on_colab else Path(__file__).parents[2]
        self.processed_data = self.base / 'data/subDataset1'
        self.rgb_features = self.processed_data / 'all_rgbs'
        self.flow_features = self.processed_data / 'all_flows'
        
        # Validate paths
        assert self.rgb_features.exists(), f"Missing RGB features: {self.rgb_features}"
        assert self.flow_features.exists(), f"Missing Flow features: {self.flow_features}"