import chess
import numpy as np

PIECE_TYPES = [chess.PAWN, chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN, chess.KING]

def board_to_tensor(board: chess.Board):
    # Encode board into a 12x8x8
    # White: [P,N,B,R,Q,K], Black: [P,N,B,R,Q,K]
    planes = np.zeros((12,8,8), dtype=np.float32)
    piece_map = board.piece_map()
    for sq, piece in piece_map.items():
        row = 7 - (sq // 8)
        col = sq % 8
        offset = 0 if piece.color == chess.WHITE else 6
        p_type = piece.piece_type
        planes[(p_type-1)+offset, row, col] = 1.0
    return planes

def move_to_index(move: chess.Move):
    return move.from_square * 64 + move.to_square

def legal_moves_mask(board: chess.Board):
    mask = np.zeros(64*64, dtype=np.float32)
    for m in board.legal_moves:
        idx = move_to_index(m)
        mask[idx] = 1.0
    return mask
