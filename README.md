AlphaZero-Lusheeta
===
- Except Minimax.py and Play.py, this code has been run and test succefully on 4100 matches in the dataset.
- dataset: get ~8000 matches from https://www.ficsgames./ with Elo > 2000

Create environment
```
conda env create -f env.yml
```

Play aganist minimax || supervised learning:
```
python battle_field.py
```
Supervised Learning
===
Process data:
```
python -m sl_data_processing.process_data
```

Encode data:
```
python -m sl_data_processing.encode_data
```
AlphaZero
===
Self-play and training (change run.py)
```
python run.py
```
Play against alphazero
```
python battle_field.py
```
