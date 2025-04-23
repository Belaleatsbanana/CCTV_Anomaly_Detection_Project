import torch
import torch.nn as nn

class Learner(nn.Module):
    def __init__(self, input_dim=2048):
        super(Learner, self).__init__()
        self.classifier = nn.Sequential(
            nn.Linear(input_dim, 1024),
            nn.LayerNorm(1024),  # Added layer normalization
            nn.LeakyReLU(0.1),    # Better gradient flow
            nn.Dropout(0.5),      # Adjusted dropout rate
            
            nn.Linear(1024, 256),
            nn.LayerNorm(256),    # Intermediate normalization
            nn.LeakyReLU(0.1),
            nn.Dropout(0.5),
            
            nn.Linear(256, 1),
            nn.Sigmoid()
        )
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                # Proper initialization for LeakyReLU
                nn.init.kaiming_normal_(m.weight, 
                                       nonlinearity='leaky_relu',
                                       a=0.1)  # Matches LeakyReLU slope
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        return self.classifier(x)