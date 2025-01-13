import pandas as pd
import chess
import numpy as np
import pandas as pd
import os
import sys
from solutions import PolynomialMovesSolution, StaticFeaturesSolution, StockfishSolution
from system import TEST_DATA_PATH, SUBMISSION_PATH
# Add data folder to pythonpath
sys.path.append(os.path.join(os.path.dirname(__file__), 'data'))

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

# MSE SCORE: 112_157
def static_features_prediction(row):
    """Make predictions using the trained StaticFeaturesSolution model"""
    # Load model if not already loaded
    if not hasattr(static_features_prediction, 'model'):
        static_features_prediction.model = StaticFeaturesSolution()
        static_features_prediction.model.load()
    
    # Make float prediction
    float_prediction = static_features_prediction.model.predict_single(row)
    
    # Round to nearest multiple of 5 and clip to valid range
    return round(float_prediction)

def stockfish_prediction(row):
    """Make predictions using the trained StockfishSolution model"""
    # Load model if not already loaded
    if not hasattr(stockfish_prediction, 'model'):
        stockfish_prediction.model = StockfishSolution()
        stockfish_prediction.model.load()
    
    float_prediction = stockfish_prediction.model.predict_single(row)
    return round(float_prediction)

def create_submission(evaluation_method, evaluation_method_name):
    test_df = pd.read_csv(TEST_DATA_PATH)
    
    # Generate predictions
    predictions = []
    for _, row in test_df.iterrows():
        pred = evaluation_method(row)
        pred = max(400, min(3330, pred))
        predictions.append(pred)
    
    
    with open(f'{SUBMISSION_PATH}/submission_{evaluation_method_name}.txt', 'w') as f:
        for pred in predictions:
            f.write(f"{pred}\n")
    
    print(f"Created submission file with {len(predictions)} predictions using {evaluation_method_name}")

if __name__ == "__main__":
    create_submission(stockfish_prediction, 'stockfish_features')
