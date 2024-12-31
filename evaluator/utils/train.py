import torch
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm  # Pour afficher la barre de progression
from .loss import AdaptiveMSELoss  # Assurez-vous d'importer la classe

# Fonction pour calculer la Quantile Loss
def quantile_loss(predictions, targets, tau):
    """
    Compute the Quantile Loss for a given quantile tau.
    Args:
        predictions (torch.Tensor): Predicted values.
        targets (torch.Tensor): True values.
        tau (float): Quantile to compute the loss for (e.g., 0.95).
    Returns:
        torch.Tensor: The computed quantile loss.
    """
    errors = targets - predictions
    return torch.mean(torch.max(tau * errors, (tau - 1) * errors))


def train_and_save_model(model, inputboard, inputmeta, data_labels, epochs=10, batch_size=8192, checkpoint_dir='../checkpoints', lambda_factor=1.0):
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
    # loss_fn = AdaptiveMSELoss(lambda_factor=lambda_factor)
    loss_fn_estimation = nn.MSELoss()

    # Entraînement
    print("Lancement de l'entraînement : ")
    for epoch in range(epochs):
        model.train()
        running_loss_estimation = 0.0
        total_absolute_distance = 0.0
        total_absolute_distance_small_labels = 0.0
        count_small_labels = 0
        total_ecart=0

        with tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}", unit="batch") as progress_bar:
            for batch, (inputs1, inputs2, labels) in enumerate(progress_bar):
                optimizer.zero_grad()

                # Le modèle retourne trois valeurs
                evaluation=model(inputs1, inputs2)
                estimation, estimation_95, estimation_05 = evaluation[:, 0:1], evaluation[:, 1:2], evaluation[:, 2:3]

                # Assurez-vous que les labels sont correctement dimensionnés
                labels = labels.unsqueeze(dim=1)  # (batch_size, 1)

                # Calcul de la perte principale (MSE pour l'estimation)
                loss_estimation = loss_fn_estimation(estimation, labels)

                # Calcul des pertes quantiles
                loss_quantile_95 = quantile_loss(estimation_95, labels, tau=0.95)
                loss_quantile_05 = quantile_loss(estimation_05, labels, tau=0.05)

                # Combiner les pertes (avec des poids si nécessaire)
                total_loss = loss_estimation + loss_quantile_95 + loss_quantile_05

                # Backpropagation
                total_loss.backward()
                optimizer.step()

                # Suivi de la perte
                running_loss_estimation += loss_estimation.item()

                # Calcul des distances absolues
                absolute_distance = torch.abs(estimation - labels).mean().item()
                total_absolute_distance += absolute_distance
                
                total_ecart+=torch.mean(loss_quantile_95-loss_quantile_05).item()

                # Calcul des distances pour les labels ayant une valeur absolue inférieure à 200
                small_labels_mask = torch.abs(labels) < 100
                if small_labels_mask.sum() > 0:  # Vérifie s'il y a des labels à faible valeur absolue
                    small_labels_distance = torch.abs(estimation[small_labels_mask] - labels[small_labels_mask]).sum().item()
                    total_absolute_distance_small_labels += small_labels_distance
                    count_small_labels += small_labels_mask.sum().item()

                # Mise à jour de la barre de progression avec les nouvelles métriques
                progress_bar.set_postfix(
                    loss=running_loss_estimation / (progress_bar.n + 1),
                    MAD=total_absolute_distance / (progress_bar.n + 1),
                    MAD_smallcase=total_absolute_distance_small_labels / (count_small_labels if count_small_labels > 0 else 1), 
                    ecart=total_ecart / (progress_bar.n + 1)
                )

                if batch%100==99:
                    torch.save(model.state_dict(), f"{checkpoint_dir}/model_epoch_{epoch+1}.pt")
        # Affichage de la perte et des métriques par époque
        print(f"Epoch {epoch+1}, Loss: {running_loss_estimation / len(dataloader)}")
        print(f"Epoch {epoch+1}, Absolute Distance Mean: {total_absolute_distance / len(dataloader)}")
        print(f"Epoch {epoch+1}, Absolute Distance Mean (small labels): {total_absolute_distance_small_labels / (count_small_labels if count_small_labels > 0 else 1)}")
        print(f"Epoch {epoch+1}, Ecart entre les quatiles: {total_ecart / len(dataloader)}")

        # Sauvegarder le modèle
        torch.save(model.state_dict(), f"{checkpoint_dir}/model_epoch_{epoch+1}.pt")

    print("Entraînement terminé et modèle sauvegardé.")
