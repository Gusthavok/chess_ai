import pandas as pd
import numpy as np
import chess
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor
import os

# Diviser le DataFrame en sous-ensembles pour un traitement parallèle
def parallelize_dataframe(df, func, num_processes=None):
    num_processes = num_processes or os.cpu_count()
    df_split = np.array_split(df, num_processes)
    with ProcessPoolExecutor(max_workers=num_processes) as executor:
        results = list(tqdm(executor.map(func, df_split), total=len(df_split), desc="Processing"))
    return pd.concat(results)

# Fonction pour traiter l'état de l'échiquier à partir de la notation FEN
def boardstate(fen):
    board = chess.Board(fen[0])
    fstr = str(fen[0])

    # Récupération des droits de roque et de l'état de mise en échec
    WCKI = 1 if board.has_kingside_castling_rights(chess.WHITE) else 0
    WCQ = 1 if board.has_queenside_castling_rights(chess.WHITE) else 0
    WCH = 1 if board.is_check() else 0
    BCKI = 1 if board.has_kingside_castling_rights(chess.BLACK) else 0
    BCQ = 1 if board.has_queenside_castling_rights(chess.BLACK) else 0
    BCH = 1 if board.was_into_check() else 0

    # Liste des informations de l'échiquier
    fw = [WCKI, WCQ, WCH]
    fb = [BCKI, BCQ, BCH]

    bstr = str(board).replace("p", "\ -1").replace("n", "\ -3").replace("b", "\ -4").replace("r", "\ -5")\
                     .replace("q", "\ -9").replace("k", "\ -100").replace("P", "\ 1").replace("N", "\ 3")\
                     .replace("B", "\ 4").replace("R", "\ 5").replace("Q", "\ 9").replace("K", "\ 100")\
                     .replace(".", "\ 0").replace("\ ", ",").replace("'", " ").replace("\n", "").replace(" ", "")[1:]
    
    bstr = list(eval(bstr))
    if "w" not in fstr:
        bstr = [-x for x in bstr[::-1]]
        fw, fb = fb, fw

    BITBOARD = fw + fb + bstr
    return BITBOARD


# Fonction pour ajuster les évaluations
def strfix(fen, tr):
    fstr = str(fen)
    
    if '#' in str(tr):
        t = -32000 if '-' in tr else 32000
    elif '\ufeff+23' in str(tr):
        t = 0
    else:
        t = int(tr)
    
    if "w" not in fstr:
        t = -t

    t = min(3200, t) / 10  # Normalisation de l'évaluation
    t = max(-3200, t)
    return t

# utile pour le multiprocessing dans load_data
def process_labels_chunk(chunk):
    return chunk.apply(lambda x: strfix(x['col1'], x['col2']), axis=1)

def process_features_chunk(chunk):
    return chunk.apply(boardstate, axis=1)

def process_features_chunk_2(chunk):
    return chunk.apply(pd.Series)

def load_data(file_path, num_rows=5000000):
    """
    Charge et prépare les données pour l'entraînement.

    Parameters:
    - file_path : Chemin vers le fichier CSV des données.
    - num_rows : Nombre de lignes à charger du CSV.

    Returns:
    - inputboard : Données d'entrée (plateaux de jeu).
    - inputmeta : Données d'entrée supplémentaires.
    - data_labels : Évaluations cibles.
    """
    # Chargement du fichier CSV
    data = pd.read_csv(file_path)

    # Traitement des features
    label_columns = [1]
    data_features = data.drop(columns=data.iloc[:, label_columns])
    data_features = data_features.head(num_rows)

    # Traitement des labels
    data_labels = data
    data_labels.columns = ['col1', 'col2']
    data_labels = data_labels.head(num_rows)
    data_labels = data_labels.astype(str)
    
    # Traitement parallèle des labels
    tqdm.pandas(desc="Processing labels")
    data_labels = parallelize_dataframe(data_labels, process_labels_chunk)

    # Traitement parallèle des features (boardstate)
    tqdm.pandas(desc="Processing features 1/2")
    data_features = parallelize_dataframe(data_features, process_features_chunk)

    # Deuxième étape de transformation des features
    tqdm.pandas(desc="Processing features 2/2")
    data_features = parallelize_dataframe(data_features, process_features_chunk_2)

    # Sélection des colonnes d'entrée (inputmeta)
    input2_columns = [0, 1, 2, 3, 4, 5]
    inputboard = data_features.drop(columns=data_features.iloc[:, input2_columns]).to_numpy()
    inputmeta = data_features.iloc[:, input2_columns].to_numpy()

    return inputboard, inputmeta, data_labels
