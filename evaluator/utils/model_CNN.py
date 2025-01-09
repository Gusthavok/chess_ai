import torch
import torch.nn as nn
import torch.nn.functional as F

import torch
import torch.nn as nn
import numpy as np
import torch
import torch.nn as nn
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

class ChessModelCNN(nn.Module):
    def __init__(self):
        super(ChessModelCNN, self).__init__()
        
        # Convolution Layer 1: Entrée (1, 8, 8) --> Sortie (16, 8, 8)
        self.conv1 = nn.Conv2d(
            in_channels=1,    # Une seule entrée (le tableau 8x8)
            out_channels=16,  # 16 cartes de caractéristiques
            kernel_size=15,    # Taille du kernel : 3x3
            padding=7         # Padding pour conserver la taille 8x8
        )
        
        # Convolution Layer 2: Entrée (16, 8, 8) --> Sortie (32, 8, 8)
        self.conv2 = nn.Conv2d(
            in_channels=16, 
            out_channels=32, 
            kernel_size=15, 
            padding=7
        )
        
        # Fully Connected Layer: Transformation en sortie finale
        self.fc1 = nn.Linear(
            in_features=32 * 8 * 8,  # Chaque case contribue à 32 features
            out_features=128         # Nombre de neurones dans la couche cachée
        )
        self.fc2 = nn.Linear(128, 3) 
        
    def forward(self, x):
        # Convolution + Activation ReLU + Pooling
        x = F.relu(self.conv1(x))       # (16, 8, 8)
        x = F.relu(self.conv2(x))       # (32, 8, 8)
        
        # Aplatir pour les couches fully connected
        x = x.view(x.size(0), -1)       # (batch_size, 32 * 8 * 8)
        
        # Fully Connected Layers
        x = F.relu(self.fc1(x))         # (batch_size, 128)
        x = self.fc2(x)                 # (batch_size, 64)
        
        return x
