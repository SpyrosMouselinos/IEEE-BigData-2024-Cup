from abc import ABC, abstractmethod
import pandas as pd
from typing import Union, Dict
import joblib
import os
import sys
from tqdm import tqdm
import time
from sklearn.metrics import mean_squared_error
import chess
from filters import filter_rating, filter_rating_deviation

class Solution(ABC):
    """Abstract base class for chess puzzle rating prediction solutions."""
    
    def __init__(self):
        self.model = None
        self.pipeline = None
        # Get the directory where solutions.py is located
        base_dir = os.path.dirname(os.path.abspath(__file__))
        # Create models directory in the same directory as solutions.py
        self.models_dir = os.path.join(base_dir, 'models')
        self.model_path = os.path.join(self.models_dir, self.__class__.__name__ + '.joblib')
        os.makedirs(self.models_dir, exist_ok=True)
    
    @abstractmethod
    def train(self, train_df: pd.DataFrame) -> None:
        """Train the model on the given training data with cross-validation."""
        pass
    
    @abstractmethod
    def predict_single(self, row: pd.Series) -> float:
        """Predict rating for a single puzzle."""
        pass
    
    @abstractmethod
    def load(self) -> None:
        """Load a pre-trained model."""
        pass
    
    def save(self) -> None:
        """Save the trained model."""
        print(f"Saving model to: {self.model_path}")
        if self.model is not None:
            joblib.dump(self.model, self.model_path)
        elif self.pipeline is not None:
            joblib.dump(self.pipeline, self.model_path)
        else:
            raise ValueError("No model to save. Train the model first.")
    
    def predict(self, data: Union[pd.DataFrame, pd.Series]) -> Union[float, list[float]]:
        """Predict ratings for one or multiple puzzles."""
        if isinstance(data, pd.Series):
            return self.predict_single(data)
        else:
            return [self.predict_single(row) for _, row in data.iterrows()]

class PolynomialMovesSolution(Solution):
    def __init__(self):
        super().__init__()
        from sklearn.pipeline import Pipeline
        from sklearn.linear_model import Ridge
        from sklearn.preprocessing import StandardScaler
        
        self.pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('ridge', Ridge())
        ])
        
    def _create_features(self, moves_series: Union[pd.Series, str]) -> pd.DataFrame:
        """Create polynomial features from moves more efficiently."""

        # Two cases:
        # 1. A whole pd.Series of moves
        # 2. A single move string
        if isinstance(moves_series, pd.Series):
            moves_count = moves_series.str.count(' ') + 1
        else:
            moves_count = pd.Series([moves_series.count(' ') + 1])
        
        return pd.DataFrame({
            'moves_inv_square': 1 / (moves_count ** 2),
            'moves_inv': 1 / moves_count,
            'moves': moves_count,
            'moves_square': moves_count ** 2
        })
    
    def train(self, train_df: pd.DataFrame) -> None:
        """Train model using cross-validation and grid search."""
        from sklearn.model_selection import GridSearchCV
        from sklearn.model_selection import KFold
       
        # Create features more efficiently
        X = self._create_features(train_df['Moves'])
        y = train_df['Rating']
        
        # Define parameter grid
        param_grid = {
            'ridge__alpha': [0.001, 0.005, 0.01, 0.05, 0.1, 1, 5, 10],
        }
        
        # Set up cross-validation
        cv = KFold(n_splits=5, shuffle=True, random_state=42)
        
        # Perform grid search
        grid_search = GridSearchCV(
            self.pipeline,
            param_grid,
            cv=cv,
            scoring='neg_mean_squared_error',
            n_jobs=-1,
            verbose=1
        )
        
        # Fit the model with progress tracking
        print("Starting grid search...")
        start_time = time.time()
        grid_search.fit(X, y)
        elapsed_time = time.time() - start_time
        print(f"Training completed in {elapsed_time:.2f} seconds")
        
        # Print best parameters
        print(f"Best parameters: {grid_search.best_params_}")
        print(f"Best cross-validation score: {-grid_search.best_score_:.2f} MSE")
        
        # After grid search, train on full dataset with best parameters
        print("Training final model on full dataset...")
        self.pipeline.set_params(**grid_search.best_params_)
        X = self._create_features(train_df['Moves'])
        y = train_df['Rating']
        self.pipeline.fit(X, y)
        
        # Print final training score
        final_mse = mean_squared_error(y, self.pipeline.predict(X))
        print(f"Final MSE on full dataset: {final_mse:.2f}")
        
        # Save the final model
        self.model = self.pipeline
        self.save()
    
    def predict_single(self, row: pd.Series) -> float:
        """Predict rating for a single puzzle."""
        if self.pipeline is None:
            raise ValueError("Model not trained or loaded. Call train() or load() first.")
        
        X = self._create_features(row['Moves'])
        return float(self.pipeline.predict(X)[0])
    
    def load(self) -> None:
        """Load pre-trained model."""
        if os.path.exists(self.model_path):
            self.pipeline = joblib.load(self.model_path)
        else:
            raise FileNotFoundError(f"No saved model found at {self.model_path}")

class StaticFeaturesSolution(Solution):
    def __init__(self):
        super().__init__()
        from sklearn.pipeline import Pipeline
        from xgboost import XGBRegressor
        
        self.pipeline = Pipeline([
            ('xgb', XGBRegressor(
                n_estimators=500,
                learning_rate=0.05,
                random_state=42,
                max_depth=12,
                colsample_bytree=0.7,
                reg_alpha=10,
                tree_method='hist',   
                min_child_weight=1,
                objective='reg:squarederror'
            ))
        ])
    
    def _create_features(self, fen: str, moves: str) -> Dict:
        """Extract static features from FEN string."""
        board = chess.Board(fen)
        features = {}

        # Basic moves features
        moves_count = moves.count(' ') + 1
        next_moves = moves.split(' ')
        features.update({
            'moves_inv_square': 1 / (moves_count ** 2),
            'moves_inv': 1 / moves_count,
            'moves': moves_count,
            'moves_square': moves_count ** 2
        })

        # Player color
        features['is_player_white'] = not board.turn  
        
        # Material counts and imbalances
        pieces = {'P': 0, 'N': 0, 'B': 0, 'R': 0, 'Q': 0,
                 'p': 0, 'n': 0, 'b': 0, 'r': 0, 'q': 0}
        
        # Count pieces
        for square in chess.SQUARES:
            piece = board.piece_at(square)
            if piece:
                if piece.symbol() != 'K' and piece.symbol() != 'k':
                    pieces[piece.symbol()] += 1
        
        # Material features
        features.update({
            'white_pawns': pieces['P'],
            'white_knights': pieces['N'],
            'white_bishops': pieces['B'],
            'white_rooks': pieces['R'],
            'white_queens': pieces['Q'],
            'black_pawns': pieces['p'],
            'black_knights': pieces['n'],
            'black_bishops': pieces['b'],
            'black_rooks': pieces['r'],
            'black_queens': pieces['q'],
            'pawn_imbalance': pieces['P'] - pieces['p'],
            'knight_imbalance': pieces['N'] - pieces['n'],
            'bishop_imbalance': pieces['B'] - pieces['b'],
            'rook_imbalance': pieces['R'] - pieces['r'],
            'queen_imbalance': pieces['Q'] - pieces['q'],
        })
        
        # King safety and position features
        white_king_square = board.king(chess.WHITE)
        black_king_square = board.king(chess.BLACK)
        
        features.update({
            'white_king_in_check': int(board.is_check() and board.turn == chess.WHITE),
            'black_king_in_check': int(board.is_check() and board.turn == chess.BLACK),
            'white_king_attackers': len(board.attackers(chess.BLACK, white_king_square)) if white_king_square else 0,
            'black_king_attackers': len(board.attackers(chess.WHITE, black_king_square)) if black_king_square else 0,
            'white_king_rank': chess.square_rank(white_king_square) if white_king_square else -1,
            'white_king_file': chess.square_file(white_king_square) if white_king_square else -1,
            'black_king_rank': chess.square_rank(black_king_square) if black_king_square else -1,
            'black_king_file': chess.square_file(black_king_square) if black_king_square else -1,
        })

        # King distance features
        if white_king_square and black_king_square:
            features.update({
                'king_distance': chess.square_distance(white_king_square, black_king_square),
                'king_manhattan_distance': chess.square_manhattan_distance(white_king_square, black_king_square),
                'king_knight_distance': chess.square_knight_distance(white_king_square, black_king_square)
            })
        else:
            features.update({
                'king_distance': -1,
                'king_manhattan_distance': -1,
                'king_knight_distance': -1
            })
        
        # Center and extended center control
        center_squares = [chess.E4, chess.E5, chess.D4, chess.D5]
        extended_center = [chess.C3, chess.D3, chess.E3, chess.F3,
                         chess.C4, chess.F4,
                         chess.C5, chess.F5,
                         chess.C6, chess.D6, chess.E6, chess.F6]
        
        white_center_control = sum(1 for sq in center_squares if board.attackers(chess.WHITE, sq))
        black_center_control = sum(1 for sq in center_squares if board.attackers(chess.BLACK, sq))
        
        features.update({
            'white_center_control': white_center_control,
            'black_center_control': black_center_control,
            'center_control_imbalance': white_center_control - black_center_control,
            'white_extended_center_control': sum(1 for sq in extended_center if board.attackers(chess.WHITE, sq)),
            'black_extended_center_control': sum(1 for sq in extended_center if board.attackers(chess.BLACK, sq)),
            'pieces_in_center': sum(1 for sq in center_squares if board.piece_at(sq)),
            'pieces_in_extended_center': sum(1 for sq in extended_center if board.piece_at(sq))
        })
        
        # Pin features
        white_pinned_pieces = sum(1 for sq in chess.SQUARES if board.is_pinned(chess.WHITE, sq))
        black_pinned_pieces = sum(1 for sq in chess.SQUARES if board.is_pinned(chess.BLACK, sq))
        
        features.update({
            'white_pinned_pieces': white_pinned_pieces,
            'black_pinned_pieces': black_pinned_pieces,
            'total_pinned_pieces': white_pinned_pieces + black_pinned_pieces
        })
        
        # Development features
        white_developed_pieces = sum(1 for sq in chess.SQUARES if board.piece_at(sq) and 
                                   board.piece_at(sq).color == chess.WHITE and 
                                   board.piece_at(sq).piece_type != chess.PAWN and
                                   chess.square_rank(sq) > 1)
        
        black_developed_pieces = sum(1 for sq in chess.SQUARES if board.piece_at(sq) and 
                                   board.piece_at(sq).color == chess.BLACK and 
                                   board.piece_at(sq).piece_type != chess.PAWN and
                                   chess.square_rank(sq) < 6)
        
        features.update({
            'white_developed_pieces': white_developed_pieces,
            'black_developed_pieces': black_developed_pieces,
            'development_difference': white_developed_pieces - black_developed_pieces
        })
        
        # Pawn structure features
        white_pawns_by_file = [0] * 8
        black_pawns_by_file = [0] * 8
        
        for sq in chess.SQUARES:
            piece = board.piece_at(sq)
            if piece and piece.piece_type == chess.PAWN:
                file_idx = chess.square_file(sq)
                if piece.color == chess.WHITE:
                    white_pawns_by_file[file_idx] += 1
                else:
                    black_pawns_by_file[file_idx] += 1
        
        features.update({
            'white_doubled_pawns': sum(1 for count in white_pawns_by_file if count > 1),
            'black_doubled_pawns': sum(1 for count in black_pawns_by_file if count > 1),
            'white_isolated_pawns': sum(1 for i, count in enumerate(white_pawns_by_file) 
                                      if count > 0 and 
                                      (i == 0 or white_pawns_by_file[i-1] == 0) and 
                                      (i == 7 or white_pawns_by_file[i+1] == 0)),
            'black_isolated_pawns': sum(1 for i, count in enumerate(black_pawns_by_file)
                                      if count > 0 and 
                                      (i == 0 or black_pawns_by_file[i-1] == 0) and 
                                      (i == 7 or black_pawns_by_file[i+1] == 0))
        })
        
        # Complexity indicators
        white_attacked_pieces = 0
        black_attacked_pieces = 0
        
        for square in chess.SQUARES:
            piece = board.piece_at(square)
            if piece:
                if piece.color == chess.WHITE and board.attackers(chess.BLACK, square):
                    white_attacked_pieces += 1
                elif piece.color == chess.BLACK and board.attackers(chess.WHITE, square):
                    black_attacked_pieces += 1
        
        features.update({
            'white_attacked_pieces': white_attacked_pieces,
            'black_attacked_pieces': black_attacked_pieces,
            'total_attacked_pieces': white_attacked_pieces + black_attacked_pieces
        })
        
        # Piece mobility (count legal moves for each piece)
        white_piece_mobility = {piece_type: 0 for piece_type in ['P', 'N', 'B', 'R', 'Q', 'K']}
        black_piece_mobility = {piece_type: 0 for piece_type in ['p', 'n', 'b', 'r', 'q', 'k']}
        
        # Create a copy of the board for move generation
        board_copy = board.copy()
        
        # Count moves for white pieces
        board_copy.turn = chess.WHITE
        for square in chess.SQUARES:
            piece = board_copy.piece_at(square)
            if piece and piece.color == chess.WHITE:
                # Get all pseudo-legal moves for this piece
                moves = sum(1 for move in board_copy.pseudo_legal_moves 
                           if move.from_square == square)
                white_piece_mobility[piece.symbol().upper()] += moves
        
        # Count moves for black pieces
        board_copy.turn = chess.BLACK
        for square in chess.SQUARES:
            piece = board_copy.piece_at(square)
            if piece and piece.color == chess.BLACK:
                # Get all pseudo-legal moves for this piece
                moves = sum(1 for move in board_copy.pseudo_legal_moves 
                           if move.from_square == square)
                black_piece_mobility[piece.symbol().lower()] += moves
        
        # Add mobility features
        features.update({
            'white_pawn_mobility': white_piece_mobility['P'],
            'white_knight_mobility': white_piece_mobility['N'],
            'white_bishop_mobility': white_piece_mobility['B'],
            'white_rook_mobility': white_piece_mobility['R'],
            'white_queen_mobility': white_piece_mobility['Q'],
            'white_king_mobility': white_piece_mobility['K'],
            'black_pawn_mobility': black_piece_mobility['p'],
            'black_knight_mobility': black_piece_mobility['n'],
            'black_bishop_mobility': black_piece_mobility['b'],
            'black_rook_mobility': black_piece_mobility['r'],
            'black_queen_mobility': black_piece_mobility['q'],
            'black_king_mobility': black_piece_mobility['k'],
            'total_mobility': sum(white_piece_mobility.values()) + sum(black_piece_mobility.values())
        })
        
        # Endgame analysis features
        board_copy = board.copy()
        
        # Play out the moves
        try:
            for move_uci in next_moves:
                board_copy.push_uci(move_uci)
            
            # Analyze final position
            final_material_balance = 0
            piece_values = {'P': 1, 'N': 3, 'B': 3, 'R': 5, 'Q': 9}
            
            for square in chess.SQUARES:
                piece = board_copy.piece_at(square)
                if piece:
                    value = piece_values.get(piece.symbol().upper(), 0)
                    if piece.color == chess.WHITE:
                        final_material_balance += value
                    else:
                        final_material_balance -= value
            
            features.update({
                'ends_in_check': int(board_copy.is_check()),
                'ends_in_checkmate': int(board_copy.is_checkmate()),
                'ends_in_stalemate': int(board_copy.is_stalemate()),
                'final_material_balance': final_material_balance,
                'material_advantage_ending': int(final_material_balance > 0),
                'material_disadvantage_ending': int(final_material_balance < 0),
                'ends_with_capture': int(board_copy.is_capture(board_copy.peek()) if board_copy.move_stack else 0),
                'ends_with_promotion': int(board_copy.peek().promotion is not None if board_copy.move_stack else 0)
            })
            
        except (ValueError, IndexError) as e:
            print(f"Error in move parsing: {e}")
            # If there's any error in move parsing, set default values
            features.update({
                'ends_in_check': 0,
                'ends_in_checkmate': 0,
                'ends_in_stalemate': 0,
                'final_material_balance': 0,
                'material_advantage_ending': 0,
                'material_disadvantage_ending': 0,
                'ends_with_capture': 0,
                'ends_with_promotion': 0
            })
        
        return features
    
    def train(self, train_df: pd.DataFrame) -> None:
        """Train model using Bayesian optimization for hyperparameter tuning."""        
        print("Creating features...")
        X = pd.DataFrame([
            self._create_features(fen, moves) 
            for fen, moves in tqdm(zip(train_df['FEN'], train_df['Moves']))
        ])
        y = train_df['Rating']        
        self.pipeline.fit(X, y)
        final_mse = mean_squared_error(y, self.pipeline.predict(X))
        print(f"Final MSE on full dataset: {final_mse:.2f}")
        
        # Save the final model
        self.model = self.pipeline
        self.save()
    
    def predict_single(self, row: pd.Series) -> float:
        """Predict rating for a single puzzle."""
        if self.pipeline is None:
            raise ValueError("Model not trained or loaded. Call train() or load() first.")
        
        X = pd.DataFrame([self._create_features(row['FEN'], row['Moves'])])
        return float(self.pipeline.predict(X)[0])
    
    def load(self) -> None:
        """Load pre-trained model."""
        if os.path.exists(self.model_path):
            self.pipeline = joblib.load(self.model_path)
        else:
            raise FileNotFoundError(f"No saved model found at {self.model_path}")

if __name__ == "__main__":
    # Example training script
    train_df = pd.read_csv('C:\\Users\\Dell\\Desktop\\IEEE-BigData-2024-Cup\\data\\lichess_db_puzzle.csv')
    train_df = filter_rating(train_df, 400, 3000)
    train_df = filter_rating_deviation(train_df, 0, 200)
    train_df = train_df.sample(n=100_000, random_state=42)

    solution = StaticFeaturesSolution()
    solution.train(train_df)
    print("Model trained and saved successfully.")   