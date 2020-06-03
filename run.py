import subprocess

# DATAROOT = '/media/vutrungnghia/New Volume/ArtificialIntelligence/Dataset/reinforcement-learning'
# MODELSZOO = '/media/vutrungnghia/New Volume/ArtificialIntelligence/Models/RL'
DATAROOT = '/home/hieu123/workspace/nghia/dataset'
MODELSZOO = '/home/hieu123/workspace/nghia/models'
N_PROCESSES = 12
for iteration in range(1, 20):
    for batch in range(0, 2):
        start = 0 + N_PROCESSES * batch
        end = N_PROCESSES + N_PROCESSES * batch
        processes = []
        for i in range(start, end):
            command = [
                'taskset',
                '--cpu-list', str(i % N_PROCESSES),
                'python',
                'self_play.py',
                '--last_iter', str(iteration - 1),
                '--game_id', str(i),
                '--dataroot', DATAROOT,
                '--models', MODELSZOO,
                '--seed', str(i),
                '--n_moves', '256',
                '--n_simulations', '200']
            p = subprocess.Popen(command)
            processes.append(p)
        for process in processes:
            process.wait()
            print('==============================================')

        command = ['python',
                   'evaluate_data.py',
                   '--dataroot', DATAROOT,
                   '--last_iter', str(iteration - 1)]

        p = subprocess.Popen(command)
        p.wait()
    print('==============================================')
    command = ['taskset',
               '--cpu-list', '0,1,2,3,4,5,6,7',
               'python',
               'train.py',
               '--dataroot', DATAROOT,
               '--last_iter', str(iteration - 1),
               '--models', MODELSZOO,
               '--epochs', '10',
               '--batch_size', '200']
    p = subprocess.Popen(command)
    p.wait()
    print('Completed')
