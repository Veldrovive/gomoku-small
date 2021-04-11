from typing import List
import time

from monte_carlo_tree_search import MCTS
from game_simulator import Game
from network import Network as NNet

import torch
import numpy as np

inverse_content = {
    "b": -1,
    " ": 0,
    "w": 1
}

game = Game(8)


def get_player(path: str, net_type: str, res_blocks=6, num_channels: int = 512, max_time: float = 2, manual_ratio: float = 0):
    net_args = {
        'lr': 0.001,
        'dropout': 0.0,
        'epochs': 10,
        'batch_size': 64,
        'cuda': torch.cuda.is_available(),
        'num_channels': num_channels,
        'res_blocks': res_blocks,
        'net_type': net_type
    }
    player = CompetitionPlayer(net_args, checkpoint=path, iterations=100, manual_ratio=manual_ratio, max_time=max_time)
    return lambda board, color, get_search_count=False: player.move(board, color, get_search_count=get_search_count)


def get_best_player():
    net_args = {
        'lr': 0.001,
        'dropout': 0.0,
        'epochs': 10,
        'batch_size': 64,
        'cuda': torch.cuda.is_available(),
        'num_channels': 512,
        'res_blocks': 6,
        'net_type': 'c'
    }
    player = CompetitionPlayer(net_args, checkpoint="./checkpoints/best.pth.tar", iterations=100)
    return lambda board, color: player.move(board, color)


def get_residual_player():
    net_args = {
        'lr': 0.001,
        'dropout': 0.0,
        'epochs': 10,
        'batch_size': 64,
        'cuda': torch.cuda.is_available(),
        'num_channels': 512,
        'res_blocks': 6,
        'net_type': 'r'
    }
    player = CompetitionPlayer(net_args, checkpoint="./checkpoints/best_residual.pth.tar", iterations=100)
    return lambda board, color: player.move(board, color)


class CompetitionPlayer:
    def __init__(self, net_args, checkpoint="./checkpoints/best.pth.tar", iterations: int = 100, manual_ratio: float = 0, max_time: float = 2):
        self.checkpoint = checkpoint
        self.max_time = max_time
        self.size = game.get_board_size()[0]
        self.mcts_args = {'numMCTSSims': iterations, 'cpuct': 1.0, 'manual_ratio': manual_ratio}
        self.net_args = net_args
        self.net = NNet(game, self.net_args)
        self.net.load_checkpoint(folder="", filename=self.checkpoint)
        self.mcts = MCTS(game, self.net, self.mcts_args)

    def from_string_array(self, board_seed):
        board = np.zeros((self.size, self.size))
        for y in range(self.size):
            for x in range(self.size):
                val = inverse_content[board_seed[y][x]]
                board[y, x] = val
        return board

    def action_to_yx(self, action):
        action_y, action_x = action // self.size, action % self.size
        return action_y, action_x

    def yx_to_action(self, y, x):
        return y * self.size + x

    def move(self, board, player, get_search_count=False):
        player = 1 if player == "w" else -1
        board = self.from_string_array(board)

        canonical_board = game.get_canonical_form(board, player)
        action_probs, search_count = self.mcts.getActionProb(canonical_board, temp=0, end_time=time.time() + self.max_time)
        action = np.argmax(action_probs)
        valid_actions = game.get_valid_moves(canonical_board)

        if valid_actions[action] == 0:
            # Then the agent tried to play an invalid move.
            # Mask the probs with the valid actions and select from that.
            valid_action_probs = action_probs*valid_actions
            action = np.argmax(valid_action_probs)

        if get_search_count:
            return self.action_to_yx(action), search_count
        else:
            return self.action_to_yx(action)
