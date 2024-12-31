import chess 
import torch
# Fonction pour traiter l'état de l'échiquier à partir de la notation FEN
def chess_to_data(board:chess.Board, as_tensor=False):

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
    if board.turn == chess.BLACK:
        bstr = [-x for x in bstr[::-1]]
        fw, fb = fb, fw

    input2 = fw + fb
    if as_tensor:
        bstr = torch.tensor(bstr, dtype=torch.float32)
        input2 = torch.tensor(input2, dtype=torch.float32)
    return (bstr, input2)