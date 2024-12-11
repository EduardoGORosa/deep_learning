import copy
import math
import random
import numpy as np
import torch
import torch.nn.functional as F
import chess
from utils import board_to_tensor, move_to_index, legal_moves_mask

def run_mcts(board: chess.Board, model, num_simulations=50, c_puct=1.0):
    # Very simplified MCTS that re-computes from root each simulation.
    # Stores stats in dictionaries keyed by FEN.

    N = {}  # visit counts
    W = {}
    Q = {}
    P = {}

    def board_key(b):
        return b.fen()

    def next_board(b, move):
        tmp = b.copy()
        tmp.push(move)
        return tmp

    def expand(b):
        key = board_key(b)
        if key in P:
            return
        obs = board_to_tensor(b)
        obs_t = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            policy_logits, value = model(obs_t)
        policy = F.softmax(policy_logits, dim=1).squeeze(0).numpy()

        mask = legal_moves_mask(b)
        policy *= mask
        if policy.sum() > 0:
            policy /= policy.sum()
        else:
            # no moves or terminal
            policy = mask

        P[key] = policy
        N[key] = 0
        W[key] = 0
        Q[key] = 0

    def simulate(b):
        path = []
        current_board = b.copy()

        # Selection
        while True:
            key = board_key(current_board)
            expand(current_board)
            if current_board.is_game_over():
                break
            moves = list(current_board.legal_moves)
            if len(moves) == 0:
                break
            best_score = -float('inf')
            best_move = None

            # UCB selection
            for m in moves:
                a = move_to_index(m)
                # Child state key
                nb = next_board(current_board, m)
                nk = board_key(nb)
                nN = N.get(nk, 0)
                # UCB calculation
                u = Q[key] + c_puct * P[key][a] * math.sqrt(N[key] + 1e-8)/(1+nN)
                if u > best_score:
                    best_score = u
                    best_move = m

            if best_move is None:
                best_move = random.choice(moves)

            path.append((current_board.copy(), best_move))
            current_board.push(best_move)
            if current_board.is_game_over():
                expand(current_board)
                break

        # Evaluate final state
        if current_board.is_game_over():
            result = current_board.result()
            if result == '1-0':
                v = 1.0
            elif result == '0-1':
                v = -1.0
            else:
                v = 0.0
        else:
            # Evaluate with network
            obs = board_to_tensor(current_board)
            obs_t = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
            with torch.no_grad():
                _, value = model(obs_t)
            v = value.item()

        # Backpropagate
        # Alternate sign for each step, assuming first step is perspective of root player.
        for i, (st, mv) in enumerate(reversed(path)):
            sign = 1 if i % 2 == 0 else -1
            skey = board_key(st)
            N[skey] += 1
            W[skey] += sign * v
            Q[skey] = W[skey]/N[skey]

    root_board = board.copy()
    expand(root_board)
    for _ in range(num_simulations):
        simulate(root_board)

    # Return final policy
    visits = np.zeros(4096, dtype=np.float32)
    moves = list(root_board.legal_moves)
    if len(moves) == 0:
        return visits
    visit_sum = 0
    for m in moves:
        a = move_to_index(m)
        nb = next_board(root_board, m)
        nk = board_key(nb)
        visits[a] = N.get(nk, 0)
        visit_sum += visits[a]

    if visit_sum > 0:
        visits /= visit_sum
    return visits
