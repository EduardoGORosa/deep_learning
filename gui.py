import tkinter as tk
from tkinter import messagebox
import chess
import numpy as np
import os
import random
from PIL import Image, ImageTk

from mcts import run_mcts
from utils import move_to_index

SQUARE_SIZE = 64
BOARD_SIZE = SQUARE_SIZE * 8
PIECE_NAMES = {
    'P': 'P', 'N': 'N', 'B': 'B', 'R': 'R', 'Q': 'Q', 'K': 'K',
    'p': 'P', 'n': 'N', 'b': 'B', 'r': 'R', 'q': 'Q', 'k': 'K'
}

class ChessGUI:
    def __init__(self, root, model):
        self.root = root
        self.root.title("AlphaZero Chess (PyTorch)")

        self.model = model
        self.board = chess.Board()
        self.move_from = None

        self.canvas = tk.Canvas(root, width=BOARD_SIZE, height=BOARD_SIZE)
        self.canvas.pack()
        self.canvas.bind("<Button-1>", self.on_click)

        self.piece_images = {}
        self.load_images()
        self.draw_board()
        self.draw_pieces()

    def load_images(self):
        piece_types = ['P','N','B','R','Q','K']
        colors = ['w','b']
        for c in colors:
            for p in piece_types:
                filename = f"images/{c}{p}.png"
                if not os.path.exists(filename):
                    print("Image not found:", filename)
                else:
                    img = Image.open(filename)
                    img = img.resize((SQUARE_SIZE, SQUARE_SIZE), Image.Resampling.LANCZOS)
                    img_tk = ImageTk.PhotoImage(img)
                    self.piece_images[c+p] = img_tk
        print("Loaded images:", self.piece_images.keys())

    def draw_board(self):
        self.canvas.delete("square")
        colors = ["#EEEED2", "#769656"]
        for row in range(8):
            for col in range(8):
                color = colors[(row+col) % 2]
                x1 = col * SQUARE_SIZE
                y1 = (7-row) * SQUARE_SIZE
                self.canvas.create_rectangle(x1, y1, x1+SQUARE_SIZE, y1+SQUARE_SIZE, fill=color, tags="square")
        self.canvas.tag_lower("square")

    def draw_pieces(self):
        self.canvas.delete("piece")
        for square in chess.SQUARES:
            piece = self.board.piece_at(square)
            if piece:
                symbol = piece.symbol()
                color = 'w' if piece.color else 'b'
                ptype = PIECE_NAMES[symbol]
                img_key = color+ptype
                if img_key in self.piece_images:
                    col = square % 8
                    row = square // 8
                    x = col * SQUARE_SIZE
                    y = (7-row)*SQUARE_SIZE
                    self.canvas.create_image(x+32, y+32, image=self.piece_images[img_key], tags="piece")

    def on_click(self, event):
        col = event.x // SQUARE_SIZE
        row = 7 - (event.y // SQUARE_SIZE)
        clicked_square = chess.square(col, row)

        if self.move_from is None:
            piece = self.board.piece_at(clicked_square)
            if piece and piece.color == self.board.turn:
                self.move_from = clicked_square
        else:
            move = chess.Move(self.move_from, clicked_square)
            if move in self.board.legal_moves:
                self.board.push(move)
                self.move_from = None
                self.draw_board()
                self.draw_pieces()
                self.root.after(100, self.ai_move)
            else:
                self.move_from = None
                self.draw_board()
                self.draw_pieces()

    def ai_move(self):
        if self.board.is_game_over():
            result = self.board.result()
            messagebox.showinfo("Game Over", f"Result: {result}")
            return

        # Run MCTS
        policy = run_mcts(self.board, self.model, num_simulations=100)
        legal_moves = list(self.board.legal_moves)
        if len(legal_moves) == 0:
            result = self.board.result()
            messagebox.showinfo("Game Over", f"Result: {result}")
            return
        legal_indices = [move_to_index(m) for m in legal_moves]
        p = policy[legal_indices]
        if p.sum() == 0:
            chosen_move = random.choice(legal_moves)
        else:
            # pick move with max visit count
            chosen_move = legal_moves[np.argmax(p)]
        self.board.push(chosen_move)
        self.draw_board()
        self.draw_pieces()

        if self.board.is_game_over():
            result = self.board.result()
            messagebox.showinfo("Game Over", f"Result: {result}")
