import chess
import torch
from ..evaluator.utils.transform_data import chess_to_data

def agent(board : chess.Board, net, as_matrix=False):
    legal_moves = list(board.legal_moves)

    resulting_boards_data_input1 = []
    resulting_boards_data_input2 = []
    
    for move in legal_moves:
        board.push(move)
        input1, input2 = chess_to_data(board, as_tensor=True, as_matrix=as_matrix)
        resulting_boards_data_input1.append(input1.unsqueeze(0))
        resulting_boards_data_input2.append(input2.unsqueeze(0))
        board.pop()

    resulting_boards_data_input1 = torch.cat(resulting_boards_data_input1, dim=0)
    resulting_boards_data_input2 = torch.cat(resulting_boards_data_input2, dim=0)
    
    scores = net(resulting_boards_data_input1, resulting_boards_data_input2)[:, 0]
    indice_move=torch.argmax(scores)
        
    move = legal_moves[indice_move]

    return move