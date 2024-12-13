import torch
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import torch.nn as nn
from tqdm import tqdm  # Pour afficher la barre de progression
from .loss import AdaptiveMSELoss  # Assurez-vous d'importer la classe

def train_and_save_model(model, inputboard, inputmeta, data_labels, epochs=10, batch_size=8192, checkpoint_dir='./models', lambda_factor=1.0):
    """
    Entraîne le modèle et le sauvegarde après chaque époque.
    Utilise la fonction de perte AdaptiveMSELoss.
    """
    # Convertir les données en tensors PyTorch
    inputboard_tensor = torch.tensor(inputboard, dtype=torch.float32)
    inputmeta_tensor = torch.tensor(inputmeta, dtype=torch.float32)
    data_labels_tensor = torch.tensor(data_labels.values, dtype=torch.float32)

    # Création d'un DataLoader
    dataset = TensorDataset(inputboard_tensor, inputmeta_tensor, data_labels_tensor)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Optimiseur
    optimizer = optim.Adam(model.parameters())

    # Utiliser AdaptiveMSELoss
    loss_fn = AdaptiveMSELoss(lambda_factor=lambda_factor)

    # Entraînement
    print("Lancement de l'entraînement : ")
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        total_absolute_distance = 0.0
        total_absolute_distance_small_labels = 0.0
        count_small_labels = 0

        # Utilisation de tqdm pour la barre de progression
        with tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}", unit="batch") as progress_bar:
            for batch, (inputs1, inputs2, labels) in enumerate(progress_bar):
                optimizer.zero_grad()
                outputs = model(inputs1, inputs2)
                labels = labels.unsqueeze(dim=1)  # Assurez-vous que les labels sont de forme correcte (batch_size, 1)

                # Calcul de la loss
                loss = loss_fn(outputs, labels)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()

                # Calcul des distances absolues
                absolute_distance = torch.abs(outputs - labels).mean().item()
                total_absolute_distance += absolute_distance

                # Calcul des distances pour les labels ayant une valeur absolue inférieure à 200
                small_labels_mask = torch.abs(labels) < 100
                if small_labels_mask.sum() > 0:  # Vérifie s'il y a des labels à faible valeur absolue
                    small_labels_distance = torch.abs(outputs[small_labels_mask] - labels[small_labels_mask]).sum().item()
                    total_absolute_distance_small_labels += small_labels_distance
                    count_small_labels += small_labels_mask.sum().item()

                # Mise à jour de la barre de progression avec les nouvelles métriques
                progress_bar.set_postfix(
                    loss=running_loss / (progress_bar.n + 1),
                    MAD=total_absolute_distance / (progress_bar.n + 1),
                    MAD_smallcase=total_absolute_distance_small_labels / (count_small_labels if count_small_labels > 0 else 1)
                )

                if batch%100==99:
                    torch.save(model.state_dict(), f"{checkpoint_dir}/model_epoch_{epoch+1}.pt")
        # Affichage de la perte et des métriques par époque
        print(f"Epoch {epoch+1}, Loss: {running_loss / len(dataloader)}")
        print(f"Epoch {epoch+1}, Absolute Distance Mean: {total_absolute_distance / len(dataloader)}")
        print(f"Epoch {epoch+1}, Absolute Distance Mean (small labels): {total_absolute_distance_small_labels / (count_small_labels if count_small_labels > 0 else 1)}")

        # Sauvegarder le modèle
        torch.save(model.state_dict(), f"{checkpoint_dir}/model_epoch_{epoch+1}.pt")

    print("Entraînement terminé et modèle sauvegardé.")
