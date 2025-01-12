# IEEE-BigData-2024-Cup
Code for IEEE BigData 2024 Cup: Predicting Chess Puzzle Difficulty
[https://knowledgepit.ai/predicting-chess-puzzle-difficulty/]

---

A chess puzzle is a particular configuration of pieces on a chessboard, where the puzzle taker is instructed to assume the role of one of the players and continue the game from that position. The player has to find from one to several moves until she delivers a mate or obtains a decisive material advantage.

In the online setting, where these are often solved, the puzzle taker only moves from one side, while the puzzle publisher provides responses from the other. One such puzzle-solving service is Lichess Trainin.g

Solving puzzles is considered one of the primary ways to hone chess skills. However, currently, the only way to reliably estimate puzzle difficulty is to present it to various chess players and see if they can solve it. 

The contest aims to predict how complex a chess puzzle is by looking at the board setup and the solutions' moves. Puzzle difficulty is measured by its Glicko-2 rating calibrated on the lichess.org website. In simplified terms, it means that Lichess models the difficulty of a puzzle by assuming that every attempt at solving a puzzle is a “match.” If a user solves the puzzle correctly, she gains a puzzle rating, and the puzzle loses the rating. The opposite happens when the user doesn’t find the complete solution (partial solutions count as “losses”). Both user and puzzle ratings are initialized at 1500. More information about the Glicko rating can be found here.

Each chess puzzle is described by the initial position (using Forsyth–Edwards Notation, or FEN) and the moves included in the puzzle solution, starting with one move leading to the puzzle position and then alternating between the moves that the puzzle solver has to find and those made by the simulated “opponent.”

IEEE Big Data 2024: We will encourage the top 3 winners to submit papers describing their solutions. It is already agreed that the conference will provide the top 3 winners with free registrations. Like in previous years, the QED Software team intends to organize a workshop devoted to the competition outcomes. According to our experience, the ability to present workshop papers may be an extra incentive for participants to consider active involvement in the competition. 

The competition aims to predict the difficulty of chess puzzles based on board configurations and moves that the solution to each puzzle consists of. The difficulty level is measured as the rating on the lichess platform. The top 3 solutions will be awarded prizes. IEEE BigData 2024 Cup: Predicting Chess Puzzle Difficulty is the sixth data science competition organized in association with the IEEE International Conference on Big Data series (IEEE BigData 2024, https://www3.cs.stonybrook.edu/~ieeebigdata2024/index.html).

---

Step 1: 

Download the data from the link: https://knowledgepit.ai/predicting-chess-puzzle-difficulty/

* Save the data in the folder `/data`

* Install zstandard: `pip install zstandard`

Step 2:

* Install python-chess: `pip install python-chess`

