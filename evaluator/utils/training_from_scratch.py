import torch
import torch.nn as nn
import torch.optim as optim

import random

from collections import namedtuple, deque

Transition = namedtuple('Transition',
                        ('initial_board_input_1', 'initial_board_input_2', 'estimated_best_board_input_1', 'estimated_best_board_input_2', 'hard_score'))


class ReplayMemory(object):
    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        """Save a transition"""
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

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

def optimize(model: nn.ModuleDict, model_smooth: nn.ModuleDict, optimizer:optim.Optimizer, memory: ReplayMemory, BATCH_SIZE:int, TAU_HARD:float, only_estimation=True):
    transitions = memory.sample(BATCH_SIZE)
    batch = Transition(*zip(*transitions))

    initial_board_input1_batch = torch.cat(batch.initial_board_input_1)
    initial_board_input2_batch = torch.cat(batch.initial_board_input_2)
    estimated_best_board_input1_batch = torch.cat(batch.estimated_best_board_input_1)
    estimated_best_board_input2_batch = torch.cat(batch.estimated_best_board_input_2)

    hard_score_batch = torch.cat(batch.hard_score).unsqueeze(1)
    score_adversaire = -model_smooth(estimated_best_board_input1_batch, estimated_best_board_input2_batch)[:, 0:1] # score de l'adversaire
    
    smooth_score_adversaire = (1-TAU_HARD)*score_adversaire + TAU_HARD*hard_score_batch 
    # print("smooth_score_adversaire", smooth_score_adversaire)    

    evaluation = model(initial_board_input1_batch, initial_board_input2_batch)

    estimation, estimation_95, estimation_05 = evaluation[:, 0:1], evaluation[:, 1:2], evaluation[:, 2:3]
    # print("estimation", estimation)    

    criterion = nn.MSELoss()
    loss_estimation = criterion(estimation, smooth_score_adversaire)
    if only_estimation:
        total_loss = loss_estimation
        ecart=0
    else:
        loss_quantile_95 = quantile_loss(estimation_95, smooth_score_adversaire, tau=0.95)
        loss_quantile_05 = quantile_loss(estimation_05, smooth_score_adversaire, tau=0.05)
        total_loss = loss_estimation + loss_quantile_95 + loss_quantile_05
        ecart=torch.mean(loss_quantile_95-loss_quantile_05).item()
        
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=50)

    total_loss.backward()
    optimizer.step()
        
    absolute_distance = torch.abs(estimation - smooth_score_adversaire).mean().item()

    return loss_estimation.item(), ecart, absolute_distance
    
def optimize_checkmate_data(model: nn.ModuleDict, optimizer:optim.Optimizer, memory_checkmate: ReplayMemory, BATCH_SIZE:int, only_estimation=True):
    transitions = memory_checkmate.sample(BATCH_SIZE)
    batch = Transition(*zip(*transitions))

    initial_board_input1_batch = torch.cat(batch.initial_board_input_1)
    initial_board_input2_batch = torch.cat(batch.initial_board_input_2)


    evaluation = model(initial_board_input1_batch, initial_board_input2_batch)

    estimation, estimation_95, estimation_05 = evaluation[:, 0:1], evaluation[:, 1:2], evaluation[:, 2:3]
    
    score_adversaire = torch.cat(batch.hard_score).unsqueeze(1) # score de l'adversaire


    criterion = nn.MSELoss()
    loss_estimation = criterion(estimation, score_adversaire)
    if only_estimation:
        total_loss = loss_estimation
        ecart=0
    else:
        loss_quantile_95 = quantile_loss(estimation_95, score_adversaire, tau=0.95)
        loss_quantile_05 = quantile_loss(estimation_05, score_adversaire, tau=0.05)
        total_loss = loss_estimation + loss_quantile_95 + loss_quantile_05
        ecart=torch.mean(loss_quantile_95-loss_quantile_05).item()
        
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=50)

    total_loss.backward()
    optimizer.step()
        
    absolute_distance = torch.abs(estimation - score_adversaire).mean().item()

    return loss_estimation.item(), ecart, absolute_distance
   

def get_hard_score(board):
    return 10*torch.sum(board, dtype=torch.float32)

def soft_update(net:nn.Module, smooth_net:nn.Module, TAU:float):
    # Soft update of the target network's weights
    # θ′ ← τ θ + (1 −τ )θ′
    target_net_state_dict = smooth_net.state_dict()
    policy_net_state_dict = net.state_dict()
    for key in policy_net_state_dict:
        target_net_state_dict[key] = policy_net_state_dict[key]*TAU + target_net_state_dict[key]*(1-TAU)
    smooth_net.load_state_dict(target_net_state_dict)