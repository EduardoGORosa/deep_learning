import chess
import torch
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import random
import os
import time

from model import ChessModel    # Your model definition file
from mcts import run_mcts       # Your MCTS function file
from utils import board_to_tensor, move_to_index

############################################################
# Training Configuration and Hyperparameters
############################################################

# How many times we go through the cycle of:
# 1) Generating self-play games
# 2) Training on the collected data
NUM_ITERATIONS = 10

# How many games of self-play we generate per iteration
GAMES_PER_ITERATION = 10

# How many MCTS simulations per move. Higher = better search quality but slower.
NUM_MCTS_SIMS = 10

# Batch size and epochs for training the neural network each iteration
BATCH_SIZE = 64
EPOCHS = 5

# Learning rate for the optimizer
LR = 1e-3

# Maximum size of the replay buffer (number of states). Older data is discarded if it grows too large.
REPLAY_BUFFER_MAX_SIZE = 10000

# Path to save final and intermediate models
MODEL_CHECKPOINT_PATH = "alphazero_chess_model.pt"


############################################################
# Functions
############################################################

def play_game(model, num_mcts_sims=50):
    """
    Play a single self-play game using the current model's parameters to guide MCTS.

    Steps:
    1. Start from the initial chess position.
    2. For each move:
       - Run MCTS from the current position to get an improved policy distribution over moves.
       - Choose a move based on the improved policy (stochastic sampling).
       - Record (state, improved_policy, current_player) for training later.
    3. Once the game ends, determine the final result (win, loss, draw).
    4. Assign final values (+1/-1/0) to each recorded state based on the game outcome and which player moved.

    Returns:
        A list of (state, policy, value) for each move in the game.
        state: 12x8x8 numpy array representing the board.
        policy: length-4096 numpy array with probabilities for each possible move.
        value: a scalar in [-1,1] indicating game outcome from that state's perspective.
    """

    board = chess.Board()
    trajectory = []
    player = 1  # +1 for White, -1 for Black. White moves first.

    while not board.is_game_over():
        # Run MCTS to get the improved policy for the current position
        policy = run_mcts(board, model, num_simulations=num_mcts_sims)

        legal_moves = list(board.legal_moves)
        if len(legal_moves) == 0:
            # No legal moves, game ends. (Shouldn't happen if board.is_game_over() is False)
            break

        # Extract probabilities only for legal moves
        legal_indices = [move_to_index(m) for m in legal_moves]
        p = policy[legal_indices]

        # Normalize probabilities if sum > 0 (it should be)
        if p.sum() > 0:
            p = p / p.sum()
        else:
            # If no move has positive probability (very unlikely), pick random
            p = np.ones(len(legal_moves)) / len(legal_moves)

        # Sample a move from the distribution p
        chosen_move = np.random.choice(legal_moves, p=p)

        # Record the current state and the full 4096-action policy
        state = board_to_tensor(board)  # shape (12,8,8)
        trajectory.append((state, policy, player))

        # Make the chosen move
        board.push(chosen_move)
        # Switch player perspective
        player = -player

    # Game ended, get result
    # '1-0' means White wins, '0-1' means Black wins, '1/2-1/2' means draw
    result = board.result()
    if result == '1-0':
        game_value = 1.0
    elif result == '0-1':
        game_value = -1.0
    else:
        game_value = 0.0

    # Assign values to each recorded step
    # If step was by White (player=1), value = game_value
    # If step was by Black (player=-1), value = -game_value (opposite)
    final_data = []
    for (s, pi, p) in trajectory:
        v = game_value if p == 1 else -game_value
        final_data.append((s, pi, v))

    return final_data


def train_network(model, optimizer, states, policies, values, batch_size=64, epochs=5):
    """
    Train the model on the given dataset.

    Input:
    - states: shape (N,12,8,8)
    - policies: shape (N,4096), improved policies from MCTS
    - values: shape (N,), final game outcomes

    Procedure:
    - Shuffle the data each epoch.
    - Compute policy and value losses and backpropagate.
    - Print loss information to monitor training progress.
    """
    model.train()
    dataset_size = len(states)

    for epoch in range(epochs):
        # Shuffle data for this epoch
        perm = np.random.permutation(dataset_size)
        epoch_policy_loss = 0.0
        epoch_value_loss = 0.0
        epoch_batches = 0

        start_epoch_time = time.time()

        for i in range(0, dataset_size, batch_size):
            batch_idx = perm[i:i+batch_size]

            s_batch = torch.tensor(states[batch_idx], dtype=torch.float32)
            p_batch = torch.tensor(policies[batch_idx], dtype=torch.float32)
            v_batch = torch.tensor(values[batch_idx], dtype=torch.float32).unsqueeze(-1)

            # Forward pass: model outputs
            policy_logits, value_preds = model(s_batch)

            # Compute policy loss (cross-entropy with target distribution p_batch)
            policy_log_probs = F.log_softmax(policy_logits, dim=1)
            loss_policy = -torch.sum(p_batch * policy_log_probs) / p_batch.size(0)

            # Compute value loss (MSE between predicted and target values)
            loss_value = torch.mean((value_preds - v_batch)**2)

            # Combined loss
            loss = loss_policy + loss_value

            # Backprop
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Accumulate metrics
            epoch_policy_loss += loss_policy.item()
            epoch_value_loss += loss_value.item()
            epoch_batches += 1

        end_epoch_time = time.time()
        avg_policy_loss = epoch_policy_loss / epoch_batches
        avg_value_loss = epoch_value_loss / epoch_batches
        print(f"  Epoch {epoch+1}/{epochs}: Policy Loss={avg_policy_loss:.4f}, "
              f"Value Loss={avg_value_loss:.4f}, Time={end_epoch_time - start_epoch_time:.2f}s")


############################################################
# Main Training Loop
############################################################

def main():
    # Initialize model and optimizer
    model = ChessModel()
    optimizer = optim.Adam(model.parameters(), lr=LR)

    # Replay buffer for storing (state, policy, value) triples from many games
    replay_buffer = []

    print("Starting training process...\n")

    for iteration in range(NUM_ITERATIONS):
        print(f"=== Iteration {iteration+1}/{NUM_ITERATIONS} ===")

        # Step 1: Generate new self-play data
        new_data = []
        iteration_start = time.time()
        for g in range(GAMES_PER_ITERATION):
            # Play a single game and get the data for that game
            game_data = play_game(model, num_mcts_sims=NUM_MCTS_SIMS)
            new_data.extend(game_data)
            print(f"  Game {g+1}/{GAMES_PER_ITERATION}: {len(game_data)} moves recorded.")
        iteration_end = time.time()
        print(f"  Self-play data generation took {iteration_end - iteration_start:.2f}s")

        # Add new data to replay buffer
        replay_buffer.extend(new_data)
        if len(replay_buffer) > REPLAY_BUFFER_MAX_SIZE:
            # If buffer is too large, remove oldest data
            replay_buffer = replay_buffer[-REPLAY_BUFFER_MAX_SIZE:]
            print(f"  Replay buffer truncated to last {REPLAY_BUFFER_MAX_SIZE} samples.")

        # Prepare data for training
        states = np.array([d[0] for d in replay_buffer], dtype=np.float32)
        policies = np.array([d[1] for d in replay_buffer], dtype=np.float32)
        values = np.array([d[2] for d in replay_buffer], dtype=np.float32)

        print(f"  Training on {len(states)} samples from the replay buffer...")
        train_network(model, optimizer, states, policies, values, batch_size=BATCH_SIZE, epochs=EPOCHS)

        # Save model checkpoint after this iteration
        checkpoint_path = f"alphazero_chess_model_iter{iteration+1}.pt"
        torch.save(model.state_dict(), checkpoint_path)
        print(f"  Checkpoint saved: {checkpoint_path}\n")

    # After all iterations, save the final model
    torch.save(model.state_dict(), MODEL_CHECKPOINT_PATH)
    print("Training complete!")
    print(f"Final model saved to {MODEL_CHECKPOINT_PATH}")


if __name__ == "__main__":
    main()
