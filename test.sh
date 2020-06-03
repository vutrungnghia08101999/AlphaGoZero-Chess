python self_play.py --dataroot '/media/vutrungnghia/New Volume/ArtificialIntelligence/Dataset/reinforcement-learning' \
                    --models '/media/vutrungnghia/New Volume/ArtificialIntelligence/Models/RL' \
                    --last_iter 0 \
                    --game_id 1 \
                    --seed 0 \
                    --n_moves 10 \
                    --n_simulations 100

python train.py '--dataroot', '/media/vutrungnghia/New Volume/ArtificialIntelligence/Dataset/reinforcement-learning' \
                '--models', '/media/vutrungnghia/New Volume/ArtificialIntelligence/Models/RL' \
                '--last_iter', '0'


python self_play.py --dataroot '/home/hieu123/workspace/dataset' \
                    --models '/home/hieu123/workspace/models' \
                    --last_iter 0 \
                    --game_id 1 \
                    --seed 0 \
                    --n_moves 10 \
                    --n_simulations 100

python train.py '--dataroot', '/home/hieu123/workspace/dataset' \
                '--models', '/home/hieu123/workspace/models' \
                '--last_iter', '0'
