from utils.dataset import load_data
from utils.model import ChessModel
from utils.train import train_and_save_model

def main():
    # Charger les données
    print("Chargement des données...")
    inputboard, inputmeta, data_labels = load_data(num_rows=5000000)

    # Construire le modèle
    print("Construction du modèle...")
    model = ChessModel()

    # Entraîner et sauvegarder le modèle
    print("Entraînement du modèle...")
    train_and_save_model(model, inputboard, inputmeta, data_labels, epochs=10, batch_size=8192)

if __name__ == "__main__":
    main()
