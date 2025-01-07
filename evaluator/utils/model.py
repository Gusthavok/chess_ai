import torch
import torch.nn as nn
import torch.nn.functional as F

class EncoderLayer(nn.Module):
    def __init__(self, output_dim):
        super(EncoderLayer, self).__init__()
        self.output_dim = output_dim
        self.dense1 = nn.Linear(1, 16)
        self.dense2 = nn.Linear(16, output_dim)

    def forward(self, x):
        # Appliquer un réseau dense pour chaque élément de la séquence
        x = F.leaky_relu(self.dense1(x))
        x = F.leaky_relu(self.dense2(x))
        return x  # Résultat de forme (batch_size, 64, output_dim)

class TransformerBlock(nn.Module):
    def __init__(self, num_heads, ff_dim, dropout_rate=0.1):
        super(TransformerBlock, self).__init__()
        self.num_heads = num_heads
        self.ff_dim = ff_dim
        self.dropout_rate = dropout_rate

        # Attention multi-têtes
        self.attn = nn.MultiheadAttention(embed_dim=ff_dim, num_heads=num_heads)
        self.ff1 = nn.Linear(ff_dim, ff_dim)
        self.ff2 = nn.Linear(ff_dim, ff_dim)

    def forward(self, x):
        # Multi-head attention
        x, _ = self.attn(x, x, x)
        x = F.dropout(x, p=self.dropout_rate)
        x = x + x  # Résidu (ajout de l'entrée initiale à la sortie)

        # Feedforward
        ff_output = F.leaky_relu(self.ff1(x))
        ff_output = F.dropout(ff_output, p=self.dropout_rate)
        ff_output = self.ff2(ff_output)

        return x + ff_output  # Ajout du résidu final

class ChessModel(nn.Module):
    def __init__(self, num_transformer_blocks=4, dim_representation_piece=8):
        super(ChessModel, self).__init__()
        self.input1 = nn.Linear(64, 1)
        self.encoder_layer = EncoderLayer(output_dim=dim_representation_piece)
        
        # Transformer: plusieurs blocs
        self.transformer_blocks = nn.ModuleList([TransformerBlock(num_heads=4, ff_dim=dim_representation_piece, dropout_rate=0.1)
                                                 for _ in range(num_transformer_blocks)])

        # Couches MLP après transformation
        self.dense1 = nn.Linear(64 * dim_representation_piece + 6, 1024) # 6=dimension de input2
        self.dense2 = nn.Linear(1024, 256)
        self.dense3 = nn.Linear(256, 256)
        self.output = nn.Linear(256, 3)

    def forward(self, input1, input2):
        # Passer input1 à travers les couches
        x1 = input1.unsqueeze(dim=2)  # (batch_size, 64, 1)
        x1 = self.encoder_layer(x1)  # (batch_size, 64, dim_representation_piece)

        # Ajouter une dimension supplémentaire pour l'attention multi-têtes
        x1 = x1.permute(1, 0, 2)  # (64, batch_size, dim_representation_piece) pour l'attention

        # Appliquer plusieurs blocs transformer
        for transformer in self.transformer_blocks:
            x1 = transformer(x1)  # Chaque transformer modifie x1
            
        # Applatir la sortie de forme (batch_size, 64, dim_representation_piece) en un vecteur (batch_size, 64 * dim_representation_piece)
        x1 = x1.permute(1, 0, 2)
        x1 = x1.flatten(start_dim=1)  # (batch_size, 64 * dim_representation_piece)

        # Concaténer avec input2
        x = torch.cat((x1, input2), dim=1)  # Concaténation sur la dimension des features

        # Passer par les couches denses (MLP)
        x = F.leaky_relu(self.dense1(x))
        x = F.leaky_relu(self.dense2(x))
        x = F.leaky_relu(self.dense3(x))

        # Sortie finale
        return self.output(x)

class SimpleChessModel(nn.Module):
    def __init__(self):
        super(SimpleChessModel, self).__init__()
        self.dense1 = nn.Linear(64 + 6, 256) # 6=dimension de input2
        # self.dense2 = nn.Linear(256, 256)
        # self.dense3 = nn.Linear(256, 256)
        self.output = nn.Linear(256, 3)# Fonction pour charger un modèle sauvegardé
    
    def forward(self, input1, input2):
        x = torch.cat((input1, input2), dim=1)
        x = F.leaky_relu(self.dense1(x))
        # x = F.leaky_relu(self.dense2(x))
        # x = F.leaky_relu(self.dense3(x))
        return self.output(x)

def load_existing_model(model, model_path):
    model.load_state_dict(torch.load(model_path))
    return model
