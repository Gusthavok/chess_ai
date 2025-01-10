from utils.dataset import load_data
from utils.model_Transformer import ChessModel, SimpleChessModel, load_existing_model
from utils.model_CNN import ChessModelCNN
from utils.training_from_scratch import *
from utils.graphics import Graphics
from utils.transform_data import chess_to_data

from math import sqrt
from random import randint
import chess
import tqdm


LEARNING_RATE, BATCH_SIZE, TAU_SMOOTH_NET= 1e-3, 1024, 1e-2
GN_START, GN_PROFONDEUR_START = 100*10+int(1024*50/160), 0
GN_END, GN_PROFONDEUR_END = 8000*10+GN_START, 30
GN_PROFONDEUR_RATE = (GN_PROFONDEUR_END- GN_PROFONDEUR_START)/(GN_END-GN_START)


def get_tau(PROFONDEUR):
    if PROFONDEUR < .99:
        return (3 - sqrt(9-8*(1-PROFONDEUR)))/(2*(1-PROFONDEUR))
    elif PROFONDEUR > 1.01:
        return (-3 + sqrt(9+8*(PROFONDEUR-1)))/(2*(PROFONDEUR-1))
    else:
        return 2/3

GN_TAU_START = get_tau(GN_PROFONDEUR_START)
GN_TAU_END = get_tau(GN_PROFONDEUR_END)

def get_tau_hard_critic(game_number):
    if game_number<GN_START:
        return GN_TAU_START
    elif game_number>GN_END:
        return GN_TAU_END
    else:
        PROFONDEUR = GN_PROFONDEUR_RATE * (game_number-GN_START) + GN_PROFONDEUR_START
        return get_tau(PROFONDEUR)
         
def get_real_hard_score(board:chess.Board, board_data):
    if board.is_insufficient_material() or board.is_stalemate():
        return torch.tensor(0, dtype=torch.float32)
    elif board.is_checkmate():
        return torch.tensor(-1000, dtype=torch.float32)
    else:
        return get_hard_score(board_data)

def get_next_board(board : chess.Board, net, random=False, as_matrix=False):
    legal_moves = list(board.legal_moves)
    if not random:
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
    else:
        indice_move = randint(0, len(legal_moves)-1)
        
    move = legal_moves[indice_move]
    board.push(move)
    
    return board # en soi pas besoin car déjà modifié en place mais pour plus de clareté

        
def main(model_name):
    as_matrix=False
    if model_name=='simple':
        model_base=SimpleChessModel
    elif model_name=='CNN':
        model_base=ChessModelCNN
        as_matrix=True
    elif model_name=='Transformer':
        model_base=ChessModel
    else: 
        raise ValueError(f"{model_name} is undefined")
    
    net = model_base() #policy_net
    smooth_net = model_base() #target_net
    if True:
        model_base=load_existing_model(model=net, model_path="../output/CNN_1.0/model_number_1")
    soft_update(smooth_net=smooth_net, net=net, TAU=1) # Les deux nets partent avec les meme valeurs
    
    memory_transition = ReplayMemory(capacity=5000000)
    optimizer = optim.AdamW(net.parameters(), lr=LEARNING_RATE, amsgrad=True)
    
    graphics = Graphics()
    for game_number in tqdm.trange(1000000):
        tau_hard_critic = get_tau_hard_critic(game_number)
        board=chess.Board()
        current_board_data=chess_to_data(board, as_tensor=True, as_matrix=as_matrix)
        
        game_ended = False
        while not game_ended:

            hard_score=get_real_hard_score(board, current_board_data[0])

            board = get_next_board(board, net, random=(game_number<GN_START-100*10), as_matrix=as_matrix)
            next_board_data = chess_to_data(board, as_tensor=True, as_matrix=as_matrix)

            memory_transition.push(current_board_data[0].unsqueeze(dim=0), current_board_data[1].unsqueeze(dim=0), next_board_data[0].unsqueeze(dim=0), next_board_data[1].unsqueeze(dim=0), hard_score.unsqueeze(dim=0)) # ('initial_board', 'estimated_best_board', 'hard_score'))
            current_board_data=next_board_data
            
            
            if board.is_checkmate() or board.is_stalemate() or board.fullmove_number>80: # or board.is_insufficient_material():
                game_ended=True
            
        # print(memory_transition.memory)
        if len(memory_transition)>50*BATCH_SIZE and game_number%1==0:# and board.fullmove_number%40==0:
            loss_estimation, ecart, absolute_distance=optimize(model=net, model_smooth=smooth_net, optimizer=optimizer, memory=memory_transition, BATCH_SIZE=BATCH_SIZE, TAU_HARD=tau_hard_critic)
            graphics.add(loss_estimation, ecart, absolute_distance, tau_hard_score=tau_hard_critic)
            soft_update(net=net, smooth_net=smooth_net, TAU=TAU_SMOOTH_NET)
                
        

        if len(memory_transition)>50*BATCH_SIZE and game_number%10==0:   
            graphics.push()
            graphics.save_plot("../output/current/graph_training")
            graphics.save("../output/current/graph_training_values")
        
        if game_number%120==0:
            torch.save(smooth_net.state_dict(), "../output/current/model_number_1")
            
            
    


if __name__ == "__main__":
    main(model_name="CNN")
