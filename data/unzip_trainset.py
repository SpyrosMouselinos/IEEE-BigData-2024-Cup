import zstandard as zstd

def unzip_trainset():
    with zstd.open('lichess_db_puzzle.csv.zst', 'rb') as f:
        with open('lichess_db_puzzle.csv', 'wb') as out:
            out.write(f.read())

unzip_trainset()