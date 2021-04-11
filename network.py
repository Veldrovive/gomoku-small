import os
import torch
import numpy as np
from game_simulator import Game

from conv_network import ConvNet
from residual_network import ResiNet

class Network:
    def __init__(self, game: Game, net_args):
        self.net_args = net_args
        if net_args["net_type"] == 'r':
            self.nnet = ResiNet(game, net_args)
        else:
            self.nnet = ConvNet(game, net_args)
        self.board_x, self.board_y = game.get_board_size()
        self.action_size = game.get_action_size()

        if self.net_args["cuda"]:
            self.nnet.cuda()

    def predict(self, board):
        board = torch.FloatTensor(board.astype(np.float64))
        if self.net_args["cuda"]:
            board = board.contiguous().cuda()
        board = board.view(1, self.board_x, self.board_y)
        self.nnet.eval()
        with torch.no_grad():
            pi, v = self.nnet(board)

        return torch.exp(pi).data.cpu().numpy()[0], v.data.cpu().numpy()[0]

    def load_checkpoint(self, folder='checkpoint', filename='checkpoint.pth.tar'):
        filepath = os.path.join(folder, filename)
        if not os.path.exists(filepath):
            raise Exception("No model in path {}".format(filepath))
        map_location = None if self.net_args["cuda"] else 'cpu'
        checkpoint = torch.load(filepath, map_location=map_location)
        self.nnet.load_state_dict(checkpoint['state_dict'])