import tkinter as tk
import os
from model import load_model
from gui import ChessGUI

def main():
    # Load model
    model = load_model("alphazero_chess_model.pt")

    root = tk.Tk()
    gui = ChessGUI(root, model)
    root.mainloop()

if __name__ == "__main__":
    main()
