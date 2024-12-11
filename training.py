import chess
import torch
import torch.optim as optim
import numpy as np
import random
import os

from model import ChessModel
from mcts import run_mcts
from utils import board_to_tensor, move_to_index

# Hyperparameters
NUM_ITERATIONS = 10         # Number of iterations of self-play + training
GAMES_PER_ITERATION = 10    # How many self-play games to generate per iteration
NUM_MCTS_SIMS = 50          # MCTS simulations per move
BATCH_SIZE = 64
EPOCHS = 5
LR = 1e-3

def play_game(model, num_mcts_sims=50):
    board = chess.Board()
    trajectory = []  # (state, policy, player)
    player = 1  # assume +1 (white), -1 (black), flip with each move

    while not board.is_game_over():
        # Get MCTS policy
        policy = run_mcts(board, model, num_simulations=num_mcts_sims)
        # Sample an action proportional to policy or just pick the max
        legal_moves = list(board.legal_moves)
        if len(legal_moves) == 0:
            break
        legal_indices = [move_to_index(m) for m in legal_moves]
        p = policy[legal_indices]
        if p.sum() > 0:
            p = p / p.sum()
            chosen_move = np.random.choice(legal_moves, p=p)
        else:
            chosen_move = random.choice(legal_moves)

        # Record the state
        state = board_to_tensor(board)
        full_policy = policy  # already indexed by [4096], matching action space
        trajectory.append((state, full_policy, player))

        board.push(chosen_move)
        player = -player

    # Game is over, assign values
    # board.result() gives '1-0', '0-1', '1/2-1/2'
    result = board.result()
    if result == '1-0':
        final_value = 1.0
    elif result == '0-1':
        final_value = -1.0
    else:
        final_value = 0.0

    # For each step in trajectory:
    # If step's player == white, value = final_value
    # If step's player == black, value = -final_value (since black is the "other" side)
    data = []
    for (s, pi, p) in trajectory:
        value = final_value if p == 1 else -final_value
        data.append((s, pi, value))
    return data

def train_network(model, optimizer, states, policies, values, batch_size=64, epochs=5):
    model.train()
    dataset_size = len(states)
    for epoch in range(epochs):
        perm = np.random.permutation(dataset_size)
        for i in range(0, dataset_size, batch_size):
            batch_idx = perm[i:i+batch_size]
            s_batch = torch.tensor(states[batch_idx], dtype=torch.float32)
            p_batch = torch.tensor(policies[batch_idx], dtype=torch.float32)
            v_batch = torch.tensor(values[batch_idx], dtype=torch.float32).unsqueeze(-1)

            policy_logits, value_preds = model(s_batch)
            # Policy loss: cross-entropy
            policy_log_probs = torch.log_softmax(policy_logits, dim=1)
            loss_policy = -torch.sum(p_batch * policy_log_probs) / p_batch.size(0)

            # Value loss: MSE
            loss_value = torch.mean((value_preds - v_batch)**2)

            loss = loss_policy + loss_value

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

def main():
    model = ChessModel()
    optimizer = optim.Adam(model.parameters(), lr=LR)

    replay_buffer = []

    for iteration in range(NUM_ITERATIONS):
        # Generate self-play games
        new_data = []
        for _ in range(GAMES_PER_ITERATION):
            game_data = play_game(model, num_mcts_sims=NUM_MCTS_SIMS)
            new_data.extend(game_data)
        replay_buffer.extend(new_data)

        # To prevent the replay buffer from growing indefinitely, you could limit its size:
        # replay_buffer = replay_buffer[-10000:]  # Keep last 10k samples

        # Extract training data
        states = np.array([d[0] for d in replay_buffer], dtype=np.float32)  # shape: (N,12,8,8)
        policies = np.array([d[1] for d in replay_buffer], dtype=np.float32) # shape: (N,4096)
        values = np.array([d[2] for d in replay_buffer], dtype=np.float32)   # shape: (N,)

        # Train the model
        train_network(model, optimizer, states, policies, values, batch_size=BATCH_SIZE, epochs=EPOCHS)
        print(f"Iteration {iteration+1}/{NUM_ITERATIONS} completed.")

    # Save the model
    torch.save(model.state_dict(), "alphazero_chess_model.pt")
    print("Training complete, model saved to alphazero_chess_model.pt")

if __name__ == "__main__":
    main()
