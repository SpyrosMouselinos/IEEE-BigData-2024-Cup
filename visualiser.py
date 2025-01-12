import chess
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from typing import Optional

def plot_fen(fen:Optional[str]=None, board:Optional[chess.Board]=None):
    if fen is None and board is None:
        raise ValueError("Either fen or board must be provided")
    
    if board is None:
        board = chess.Board(fen)    
    
    _, ax = plt.subplots(figsize=(8, 8))
    
    # Draw the chess board
    for i in range(8):
        for j in range(8):
            color = 'white' if (i + j) % 2 == 0 else 'gray'
            ax.add_patch(Rectangle((j, 7-i), 1, 1, facecolor=color))
    
    # Dictionary for piece symbols
    pieces_symbols = {
        'p': '♟', 'P': '♙',
        'n': '♞', 'N': '♘',
        'b': '♝', 'B': '♗',
        'r': '♜', 'R': '♖',
        'q': '♛', 'Q': '♕',
        'k': '♚', 'K': '♔'
    }
    
    # Place pieces
    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if piece is not None:
            symbol = pieces_symbols[piece.symbol()]
            j = chess.square_file(square)
            i = 7 - chess.square_rank(square)
            ax.text(j + 0.5, i + 0.5, symbol, fontsize=30, ha='center', va='center')
    
    # Set board properties
    ax.set_xlim(0, 8)
    ax.set_ylim(0, 8)
    plt.axis('off')
    plt.show()

if __name__ == "__main__":
    plot_fen(fen="q3k1nr/1pp1nQpp/3p4/1P2p3/4P3/B1PP1b2/B5PP/5K2 b k - 0 17")