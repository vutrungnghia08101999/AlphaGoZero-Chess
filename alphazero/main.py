import torch
import torch.multiprocessing as mp

def f(d: dict, x: int):
    d[x] = -100000
    print(d)

num_processes = 4
processes = []
d = {}
# logger.info("Spawning %d processes..." % num_processes)
with torch.no_grad():
    for i in range(num_processes):
        p = mp.Process(target=f, args=(d, i))
        p.start()
        processes.append(p)
    for p in processes:
        p.join()

print("Finished multi-process MCTS!")
