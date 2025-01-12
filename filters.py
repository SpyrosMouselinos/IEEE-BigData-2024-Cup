import pandas as pd
    

# Lets make filters 
def filter_rating(df, min_rating=400, max_rating=3000):
    return df[(df['Rating'] >= min_rating) & (df['Rating'] <= max_rating)]

def filter_moves(df, min_moves=1, max_moves=12):
    return df[(df['Moves'].str.split().str.len() >= min_moves) & (df['Moves'].str.split().str.len() <= max_moves)]

def filter_rating_deviation(df, min_rating_deviation=0, max_rating_deviation=125):
    return df[(df['RatingDeviation'] >= min_rating_deviation) & (df['RatingDeviation'] <= max_rating_deviation)]

def filter_popularity(df, min_popularity=0, max_popularity=1000):
    return df[(df['Popularity'] >= min_popularity) & (df['Popularity'] <= max_popularity)]

def apply_all_filters(df):
    df = filter_rating(df)
    df = filter_moves(df)
    df = filter_rating_deviation(df)
    df = filter_popularity(df)
    return df[['PuzzleId', 'FEN', 'Moves', 'Rating']]

if __name__ == "__main__":
    df = pd.read_csv('lichess_db_puzzle.csv')
    df = df.dropna()
    df = apply_all_filters(df)
    print(df['Rating'].describe())
