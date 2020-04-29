taskset --cpu-list 0,1,2,3,4,5,6,7 python -m alphazero.self_play --last_iter 0
python -m alphazero.evaluate_data --last_iter 0
#taskset --cpu-list 0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15 python -m alphazero.training --last_iter 0

#taskset --cpu-list 0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31 python -m alphazero.self_play --last_iter 1
#python -m alphazero.evaluate_data --last_iter 1
#taskset --cpu-list 0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15 python -m alphazero.training --last_iter 1

