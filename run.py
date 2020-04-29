import subprocess


DATAROOT = '/media/vutrungnghia/New Volume/ArtificialIntelligence/Dataset/reinforcement-learning'
MODELSZOO = '/media/vutrungnghia/New Volume/ArtificialIntelligence/Models/RL'
N_GAMES = 32

for iteration in range(1, 4):
    processes = []
    for i in range(N_GAMES):
        command = [
            'taskset',
            '--cpu-list', str(i),
            'python',
            '-m', 'alphazero.self_play',
            '--last_iter', str(iteration - 1),
            '--game_id', str(i),
            '--dataroot', DATAROOT,
            '--modelszoo', MODELSZOO,
            '--n_moves', '512',
            '--n_simulation', '400']
        p = subprocess.Popen(command)
        processes.append(p)
    for process in processes:
        process.wait()
    print('==============================================')

    command = ['python',
               '-m', 'alphazero.evaluate_data',
               '--dataroot', DATAROOT,
               '--last_iter', str(iteration - 1)]

    p = subprocess.Popen(command)
    p.wait()
    print('==============================================')
    command = ['taskset',
               '--cpu-list', '0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31',
               'python',
               '-m', 'alphazero.training',
               '--dataroot', DATAROOT,
               '--last_iter', str(iteration - 1),
               '--modelszoo', MODELSZOO,
               '--epochs', '20',
               '--batch_size', '200']
    p = subprocess.Popen(command)
    p.wait()
    print('Completed')
