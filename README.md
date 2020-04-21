AlphaZero-Lusheeta
===

- chess_rules:
    - ChessObjects: Pawm, King, Queen, Rook, Bishop, Knight, Move, Spot and Board
    - BasicRules: Basic rules of pieces which include: influence spot and reached spot (haven't check is valid)
    - Rules: Implement all rules of chess
- reinforcement_learing:
    - placeholder
- supervised_learning:
    - configs.yml:
    - processed_data.py: get ~5000 matches from https://www.ficsgames./ with Elo > 2000, preprocess data (remove everything except moves) for encoding in the next step
    - encode_data.py: encode processed data with form (state, valid_actions, expect_action):
        - state: 8 x 8 x 26
        - valid_actions: 8 x 8 x 78
        - expect_action: 8 x 8 x 78
    - evaluate_toy_data:
    - utils:
- TensorBoard.py: include last state, current state and current turn
- Minimax: Minimax implementation
- Play: Play against Minimax

NOTE
===================
- Except Minimax.py and Play.py, this code has been run and test succefully on 4100 matches in the dataset.
