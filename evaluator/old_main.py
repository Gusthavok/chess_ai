from utils.dataset import load_data
from evaluator.utils.model_Transformer import ChessModel, load_existing_model
from utils.train import train_and_save_model
from utils.training_from_scratch import *

def main_for_tuning_on_dataset():
    # Construire le modèle
    print("Construction du modèle...")
    model = ChessModel()
    # model_path="../checkpoints/checkmate_and_king_capture_1/model_epoch_3.pt"
    # model=load_existing_model(model, model_path)
    
    # Charger les données
    print("Chargement des données...")
    file_path='../dataset/capture_king.csv' #'../dataset/chessData.csv'
    inputboard, inputmeta, data_labels = load_data(file_path=file_path, num_rows=5000000)



    # Entraîner et sauvegarder le modèle
    print("Entraînement du modèle...")
    train_and_save_model(model, inputboard, inputmeta, data_labels, epochs=10, batch_size=8192)

    
    

if __name__ == "__main__":
    main_for_tuning_on_dataset()
