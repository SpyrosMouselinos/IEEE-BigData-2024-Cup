import platform
import os

# Set up the path accordingly
if platform.system() == "Linux":
    OS_PATH = "/workspace/experimental/IEEE-BigData-2024-Cup"
    STOCKFISH_BIN_PATH = f"{OS_PATH}/stockfish/stockfish-linux-x86-64-avx2"
    TRAIN_DATA_PATH = f"{OS_PATH}/data/lichess_db_puzzle.csv"
    TEST_DATA_PATH = f"{OS_PATH}/data/test_data_set.csv"
    SUBMISSION_PATH = f"{OS_PATH}/submissions"
else:
    OS_PATH = "C:\\Users\\Dell\\Desktop\\IEEE-BigData-2024-Cup"
    STOCKFISH_BIN_PATH = f"{OS_PATH}\\stockfish\\stockfish-windows-x86-64-avx2.exe"
    TRAIN_DATA_PATH = f"{OS_PATH}\\data\\lichess_db_puzzle.csv"
    TEST_DATA_PATH = f"{OS_PATH}\\data\\test_data_set.csv"
    SUBMISSION_PATH = f"{OS_PATH}\\submissions"

# Create the submission path if it doesn't exist
if not os.path.exists(SUBMISSION_PATH):
    os.makedirs(SUBMISSION_PATH)
