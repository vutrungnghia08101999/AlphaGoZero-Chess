AlphaZero-Lusheeta
===
- Except Minimax.py and Play.py, this code has been run and test succefully on 4100 matches in the dataset.
- dataset: get ~5000 matches from https://www.ficsgames./ with Elo > 2000

Play aganist minimax:
```
python -m minimax.Play
```

Process data:
```
python -m sl_data_processing.process_data
```

Encode data:
```
python -m sl_data_processing.encode_data
```