import pandas as pd
import chess
import numpy as np
import pandas as pd
import os
import sys
from solutions import PolynomialMovesSolution, StaticFeaturesSolution


# Add data folder to pythonpath
sys.path.append(os.path.join(os.path.dirname(__file__), 'data'))

# Helper functions
def count_pieces(fen):
    """Count pieces for both colors from FEN"""
    board = chess.Board(fen)
    
    # Count pieces for both colors
    white_pieces = {
        'P': len(board.pieces(chess.PAWN, chess.WHITE)),
        'N': len(board.pieces(chess.KNIGHT, chess.WHITE)),
        'B': len(board.pieces(chess.BISHOP, chess.WHITE)),
        'R': len(board.pieces(chess.ROOK, chess.WHITE)),
        'Q': len(board.pieces(chess.QUEEN, chess.WHITE)),
    }
    
    black_pieces = {
        'p': len(board.pieces(chess.PAWN, chess.BLACK)),
        'n': len(board.pieces(chess.KNIGHT, chess.BLACK)),
        'b': len(board.pieces(chess.BISHOP, chess.BLACK)),
        'r': len(board.pieces(chess.ROOK, chess.BLACK)),
        'q': len(board.pieces(chess.QUEEN, chess.BLACK)),
    }
    
    return white_pieces, black_pieces

def material_difference(fen):
    """Calculate material difference (white - black)"""
    white_pieces, black_pieces = count_pieces(fen)
    
    # Standard piece values
    values = {'P': 1, 'N': 3, 'B': 3, 'R': 5, 'Q': 9}
    
    white_material = sum(count * values[piece] for piece, count in white_pieces.items())
    black_material = sum(count * values[piece.upper()] for piece, count in black_pieces.items())
    
    return white_material - black_material


# MSE SCORE: 165_286
def mean_based_prediction(row):
    base_rating = 1515
    return base_rating

# MSE SCORE: 164_044
def polynomial_moves_prediction(row):
    """Make predictions using the trained PolynomialMovesSolution model"""
    # Load model if not already loaded
    if not hasattr(polynomial_moves_prediction, 'model'):
        polynomial_moves_prediction.model = PolynomialMovesSolution()
        polynomial_moves_prediction.model.load()
    
    # Make float prediction
    float_prediction = polynomial_moves_prediction.model.predict_single(row)

    # Round to the nearest multiple of 10
    return round(float_prediction / 10) * 10

def static_features_prediction(row):
    """Make predictions using the trained StaticFeaturesSolution model"""
    # Load model if not already loaded
    if not hasattr(static_features_prediction, 'model'):
        static_features_prediction.model = StaticFeaturesSolution()
        static_features_prediction.model.load()
    
    # Make float prediction
    float_prediction = static_features_prediction.model.predict_single(row)
    
    # Round to nearest multiple of 5 and clip to valid range
    return round(float_prediction / 5) * 5

def create_submission(evaluation_method, evaluation_method_name):
    test_df = pd.read_csv('C:\\Users\\Dell\\Desktop\\IEEE-BigData-2024-Cup\\data\\test_data_set.csv')
    
    # Generate predictions
    predictions = []
    for _, row in test_df.iterrows():
        pred = evaluation_method(row)
        pred = max(400, min(3330, pred))
        predictions.append(pred)
    
    
    with open(f'../submissions/submission_{evaluation_method_name}.txt', 'w') as f:
        for pred in predictions:
            f.write(f"{pred}\n")
    
    print(f"Created submission file with {len(predictions)} predictions using {evaluation_method_name}")

def estimate_mse_score(evaluation_method, evaluation_method_name):
    """Estimate MSE score by sampling training data multiple times using stratified sampling."""
    train_df = pd.read_csv('lichess_db_puzzle.csv')
    train_df = train_df.dropna()
    
    # Create rating bins for stratification
    train_df['rating_bin'] = pd.qcut(train_df['Rating'], q=50)  # 50 quantile bins
    
    mse_scores = []
    for _ in range(5):
        # Stratified sampling
        subsampled_train_df = train_df.groupby('rating_bin', group_keys=False).apply(
            lambda x: x.sample(n=int(100_000 * len(x)/len(train_df)))
        )
        
        # Calculate predictions for the sampled data
        predictions = [evaluation_method(row) for _, row in subsampled_train_df.iterrows()]
        
        # Calculate MSE
        mse = np.mean((subsampled_train_df['Rating'] - predictions) ** 2)
        mse_scores.append(mse)
    
    print(f"MSE score for {evaluation_method_name}: {np.mean(mse_scores):.2f} ± {np.std(mse_scores):.2f}")
    return

if __name__ == "__main__":
    # Create submission with the static features model
    print("Creating submission with static features model...")
    create_submission(static_features_prediction, 'static_features')
    
    
    # You can keep the other models for comparison
    # print("\nCreating polynomial moves submission for comparison...")
    # create_submission(polynomial_moves_prediction, 'polynomial_moves')
    # estimate_mse_score(polynomial_moves_prediction, 'polynomial_moves')
