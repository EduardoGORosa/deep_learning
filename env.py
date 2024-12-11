import gymnasium as gym
import numpy as np
import chess

class ChessEnv(gym.Env):
    """
    A simplified chess environment for self-play.
    """
    metadata = {'render.modes': ['human']}
    
    def __init__(self):
        super(ChessEnv, self).__init__()
        self.board = chess.Board()
        # Action space: up to 64*64 possible moves (from_sq * 64 + to_sq)
        self.action_space = gym.spaces.Discrete(64*64)
        # Observation space: 12 planes of size 8x8
        self.observation_space = gym.spaces.Box(low=0, high=1, shape=(12,8,8), dtype=np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.board = chess.Board()
        obs = self._get_obs()
        return obs, {}

    def step(self, action: int):
        from_sq = action // 64
        to_sq = action % 64
        move = chess.Move(from_sq, to_sq)

        if move not in self.board.legal_moves:
            # Illegal move => immediate loss
            done = True
            reward = -1.0
            return self._get_obs(), reward, done, False, {}
        
        self.board.push(move)
        
        done = self.board.is_game_over()
        if done:
            result = self.board.result()
            if result == '1-0':
                # White wins
                reward = 1.0 if self.board.turn == chess.BLACK else -1.0
            elif result == '0-1':
                # Black wins
                reward = 1.0 if self.board.turn == chess.WHITE else -1.0
            else:
                # Draw
                reward = 0.0
        else:
            reward = 0.0
        
        return self._get_obs(), reward, done, False, {}
    
    def _get_obs(self):
        # Encode board into a 12x8x8 binary array
        obs = np.zeros((12,8,8), dtype=np.float32)
        piece_map = self.board.piece_map()
        for sq, piece in piece_map.items():
            row = 7 - (sq // 8)
            col = sq % 8
            offset = 0
            if piece.color == chess.BLACK:
                offset = 6
            # piece_type: P=1,N=2,B=3,R=4,Q=5,K=6
            p_type = piece.piece_type
            # Assign to correct plane
            obs[(p_type-1)+offset, row, col] = 1
        return obs

    def render(self):
        print(self.board)

    def legal_actions(self):
        # Return a mask of legal moves
        mask = np.zeros(self.action_space.n, dtype=np.float32)
        for m in self.board.legal_moves:
            idx = (m.from_square * 64) + m.to_square
            mask[idx] = 1.0
        return mask
