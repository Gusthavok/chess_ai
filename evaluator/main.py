from utils.dataset import load_data
from utils.model import ChessModel, load_existing_model
from utils.training_from_scratch import *
from utils.graphics import Graphics
from utils.transform_data import chess_to_data

import chess
import tqdm

LEARNING_RATE, BATCH_SIZE, TAU_SMOOTH_NET= 1e-3, 8192, 1e-2
GN_START, GN_TAU_START, GN_END, GN_TAU_END = 100, 1., 400, .2

GN_RATE = (GN_TAU_END - GN_TAU_START)/(GN_END-GN_START)

def get_tau_hard_critic(game_number):
    truncated_gn = game_number - GN_START
    if truncated_gn<0:
        return 1
    elif truncated_gn>GN_END:
        return GN_TAU_END
    else:
        return GN_RATE*truncated_gn
         
def get_real_hard_score(board:chess.Board, board_data):
    if board.is_insufficient_material() or board.is_stalemate():
        return torch.tensor(0, dtype=torch.float32)
    elif board.is_checkmate():
        return torch.tensor(-1000, dtype=torch.float32)
    else:
        return get_hard_score(board_data)

def get_next_board(board : chess.Board, net):

    resulting_boards_data_input1 = []
    resulting_boards_data_input2 = []
    legal_moves = list(board.legal_moves)
    
    for move in legal_moves:
        board.push(move)
        input1, input2 = chess_to_data(board)
        resulting_boards_data_input1.append(input1)
        resulting_boards_data_input2.append(input2)
        board.pop()

    resulting_boards_data_input1 = torch.tensor(resulting_boards_data_input1, dtype=torch.float32)
    resulting_boards_data_input2 = torch.tensor(resulting_boards_data_input2, dtype=torch.float32)
    
    scores = net(resulting_boards_data_input1, resulting_boards_data_input2)[:, 0]
    
    move = legal_moves[torch.argmax(scores)]
    board.push(move)
    
    return board # en soi pas besoin car déjà modifié en place mais pour plus de clareté

        
def main():
    net = ChessModel() #policy_net
    smooth_net = ChessModel() #target_net
    soft_update(smooth_net=smooth_net, net=net, TAU=1) # Les deux nets partent avec les meme valeurs
    
    memory_transition = ReplayMemory(capacity=600000)
    optimizer = optim.AdamW(net.parameters(), lr=LEARNING_RATE, amsgrad=True)
    
    graphics = Graphics()
    for game_number in tqdm.trange(1000000):
        tau_hard_critic = get_tau_hard_critic(game_number)
        board=chess.Board()
        current_board_data=chess_to_data(board, as_tensor=True)
        
        game_ended = False
        while not game_ended:

            hard_score=get_real_hard_score(board, current_board_data[1])

            board = get_next_board(board, net)
            next_board_data = chess_to_data(board, as_tensor=True)

            memory_transition.push(current_board_data[0].unsqueeze(dim=0), current_board_data[1].unsqueeze(dim=0), next_board_data[0].unsqueeze(dim=0), next_board_data[1].unsqueeze(dim=0), hard_score.unsqueeze(dim=0)) # ('initial_board', 'estimated_best_board', 'hard_score'))
            current_board_data=next_board_data
            
            
            if board.is_checkmate() or board.is_stalemate() or board.fullmove_number>80: # or board.is_insufficient_material():
                game_ended=True
            
            if len(memory_transition)>50*BATCH_SIZE and board.fullmove_number%40==0:
                loss_estimation, ecart, absolute_distance=optimize(model=net, model_smooth=smooth_net, optimizer=optimizer, memory=memory_transition, BATCH_SIZE=BATCH_SIZE, TAU_HARD=tau_hard_critic)
                graphics.add(loss_estimation, ecart, absolute_distance)
                soft_update(net=net, smooth_net=smooth_net, TAU=TAU_SMOOTH_NET)
                
        

        if len(memory_transition)>50*BATCH_SIZE and game_number%15==0:   
            graphics.push()
            graphics.save_plot("graph_training")
            graphics.save("graph_training_values")
            
            
    
    
    

if __name__ == "__main__":
    main()
