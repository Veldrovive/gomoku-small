import numpy as np
import torch
import torch.nn.functional as F

class Game:
    content = {
        -1: "b",
        0: "-",
        1: "w"
    }

    inverse_content = {
        "b": -1,
        "-": 0,
        "w": 1
    }

    def __init__(self, n):
        self.size = n
        self.case_map = {
            -1: "TooLong",
            0: "Closed",
            1: "SemiOpen",
            2: "Open"
        }
        self.cases = self.create_case_stack()
        self.MAX_SCORE = 100000

    def get_init_board(self, play_random_moves=2):
        board = np.zeros((self.size, self.size))
        if play_random_moves > 0:
            moves = np.random.choice(self.size ** 2, play_random_moves * 2, replace=False)
            for i, action in enumerate(moves):
                player = 1 if i < play_random_moves else -1
                board = self.get_next_state(board, player, action)[0]
        return board

    def get_board_size(self):
        return self.size, self.size

    def get_action_size(self):
        return self.size ** 2

    def get_next_state(self, board, player, action):
        action_y, action_x = action // self.size, action % self.size
        new_board = board.copy()
        new_board[action_y, action_x] = player
        return new_board, -player

    def get_valid_moves(self, board):
        zeros = np.where(board == 0)
        valid_moves = zeros[0] * self.size + zeros[1]
        all_moves = np.zeros(self.get_action_size(), dtype=int)
        all_moves[valid_moves] = 1
        return all_moves

    def create_case_weights(self, seq_size: int):
        weight_size = seq_size + 2
        mid = weight_size // 2 + 1
        diag_base_weight = torch.zeros((weight_size, weight_size), dtype=torch.double)
        vert_base_weight = torch.zeros((weight_size, weight_size), dtype=torch.double)
        for i in range(seq_size):
            diag_base_weight[i + 1, i + 1] = 1
            vert_base_weight[i + 1, mid] = 1
        diag_weights = [torch.clone(diag_base_weight) for _ in range(6)]
        vert_weights = [torch.clone(vert_base_weight) for _ in range(6)]
        diag_weights[0][0, 0] = 1  # Higher case 1
        vert_weights[0][0, mid] = 1
        diag_weights[1][-1, -1] = 1  # Higher case 2
        vert_weights[1][-1, mid] = 1

        diag_weights[2][0, 0] = -1  # Closed case
        diag_weights[2][-1, -1] = -1
        vert_weights[2][0, mid] = -1
        vert_weights[2][-1, mid] = -1

        diag_weights[3][0, 0] = -1  # Semi-closed case 1
        vert_weights[3][0, mid] = -1
        diag_weights[4][-1, -1] = -1  # Semi-closed case 2
        vert_weights[4][-1, mid] = -1

        diag_weight = torch.stack(diag_weights)
        diag_weight = diag_weight.view(6, 1, weight_size, weight_size)

        vert_weight = torch.stack(vert_weights)
        vert_weight = vert_weight.view(6, 1, weight_size, weight_size)
        return diag_weight, vert_weight

    def create_case_stack(self):
        case_sizes = [2, 3, 4, 5]
        return [(size, self.create_case_weights(size)) for size in case_sizes]

    def evaluate_position(self, board):
        opens = {}
        semi_opens = {}
        for i in [2, 3, 4, 5]:  # I know you can do this with range. This is more verbose don't @ me
            opens[i] = 0
            semi_opens[i] = 0
        board = torch.from_numpy(board)
        board = F.pad(board, pad=(1, 1, 1, 1), value=-1)
        board = board.view(1, 1, board.shape[-1], board.shape[-1])
        for size, weights in self.cases:
            checks = [(size + 1, -1), (size + 1, -1), (size + 2, 0), (size + 1, 1), (size + 1, 1),
                      (size, 2)]  # -1: too long, 0: closed, 1: semi-closed, 2: open
            for weight in weights:
                for rotate in [True, False]:
                    if rotate:
                        weight = torch.rot90(weight, dims=[-2, -1])
                    out = F.conv2d(board, weight)
                    out = out.view(6, out.shape[-1], out.shape[-1])
                    found_cases = {}
                    if rotate:
                        weight = torch.rot90(weight, k=-1, dims=[-2, -1])
                    for i, (count, case) in enumerate(checks):
                        # We have to check all cases so the found_cases can be built up
                        out_channel = out[i].numpy()
                        case_indices = np.where(out_channel == count)
                        for found_case in zip(*case_indices):
                            if found_case not in found_cases:
                                found_cases[found_case] = None
                                if case >= 0:
                                    # print(f"Found case of length {size} at {found_case} being {self.case_map[case]}")
                                    if case == 1:
                                        # Then we found a new semi-open
                                        semi_opens[size] += 1
                                    elif case == 2:
                                        # Then we found a new semi-open
                                        opens[size] += 1
                                else:
                                    # print(f"Disregarded case of length {size} at {found_case} being {self.case_map[case]}")
                                    pass
        return opens, semi_opens

    def get_manual_board_score(self, board):
        open_current, semi_current = self.evaluate_position(board)
        open_other, semi_other = self.evaluate_position(self.get_canonical_form(board, -1))

        if open_current[5] >= 1 or semi_current[5] >= 1:
            return self.MAX_SCORE

        elif open_other[5] >= 1 or semi_other[5] >= 1:
            return -self.MAX_SCORE

        return (-10000 * (open_other[4] + semi_other[4]) +
                500 * open_current[4] +
                50 * semi_current[4] +
                -100 * open_other[3] +
                -30 * semi_other[3] +
                50 * open_current[3] +
                10 * semi_current[3] +
                open_current[2] + semi_current[2] - open_other[2] - semi_other[2])

    def get_game_ended(self, board, player):
        # Returns -1 is black won and 1 if white won. 0 if there was a draw. None if the game is ongoing.
        def detect_five(y_start, x_start, d_y, d_x):
            seq_length = None
            seq_player = None
            y_max = self.size - 1
            x_max = self.size - 1
            while 0 <= y_start <= y_max and 0 <= x_start <= x_max:
                cur_player = board[y_start, x_start]
                if cur_player == seq_player:
                    seq_length += 1
                else:
                    if seq_length == 5:
                        return seq_player
                    seq_length = None
                    seq_player = None
                    if cur_player != 0:
                        seq_length = 1
                        seq_player = cur_player
                y_start += d_y
                x_start += d_x
            if seq_length == 5:
                return seq_player
            return None

        # Check for a winner
        winner = None
        x_start = 0
        for y_start in range(len(board)):
            # Check directions (0, 1) and (1, 1)
            winner = winner or detect_five(y_start, x_start, 0, 1)
            winner = winner or detect_five(y_start, x_start, 1, 1)
        x_start = len(board[0]) - 1
        for y_start in range(len(board)):
            # Check direction (1, -1)
            winner = winner or detect_five(y_start, x_start, 1, -1)
        y_start = 0
        for x_start in range(len(board[0])):
            # Check direction (1, 0)
            winner = winner or detect_five(y_start, x_start, 1, 0)
            if x_start > 0:
                # Check the rows that were not on the y pass
                winner = winner or detect_five(y_start, x_start, 1, 1)
            if x_start < len(board[0]) - 1:
                # Chck the rows that were not on the second y pass
                winner = winner or detect_five(y_start, x_start, 1, -1)
        if winner is not None:
            return player * winner  # TODO: Make sure this actually verifies the winner

        if np.max(self.get_valid_moves(board)) > 0:
            return 0
        return 0.01

    def get_canonical_form(self, board, player):
        return board * player

    def string_representation(self, board, highlight_action=None, include_numbers=False):
        action_y = -1 if highlight_action is None else highlight_action // self.size
        action_x = -1 if highlight_action is None else highlight_action % self.size

        def get_char(value, y, x):
            char = self.content[value]
            if action_y == y and action_x == x:
                char = char.upper()
            return char
        str_rep = '\n'.join([(f'{y} ' if include_numbers else '') + ''.join([get_char(val, y, x) for x, val in enumerate(line)]) for y, line in enumerate(board)])
        if include_numbers:
            str_rep = f"  {''.join([str(x) for x in range(self.size)])}\n{str_rep}"
        return str_rep

    def from_string(self, board_seed):
        inverse_content = {
            "b": -1,
            " ": 0,
            "w": 1
        }
        board = np.zeros((self.size, self.size))
        for y in range(self.size):
            for x in range(self.size):
                val = inverse_content[board_seed[y][x]]
                board[y, x] = val
        return board

if __name__ == "__main__":
    import gomaku_tester
    board_seed = """--------
-w------
--wb----
--bwb---
----wb--
------b-
-------b
w-------"""
    board_seed = board_seed.replace("-", " ")
    board_seed = board_seed.split("\n")
    game = Game(8)
    board = game.from_string(board_seed)
    print(game.get_game_ended(board, 1))
    print(gomaku_tester.is_win(board_seed))
