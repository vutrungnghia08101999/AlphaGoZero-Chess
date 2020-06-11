import chess

class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
# board


def compute_material_scores(board: chess.Board):
    black = {
        'p': 1,
        'n': 3,
        'b': 3,
        'r': 5,
        'q': 10,
        'k': 4
    }

    white = {
        'P': 1,
        'N': 3,
        'B': 3,
        'R': 5,
        'Q': 10,
        'K': 4
    }
    s = str(board)
    s = s.replace('.', '')
    s = s.replace('\n', '')
    s = s.replace(' ', '')
    pieces = list(s)
    black_scores = 0
    white_scores = 0
    for piece in pieces:
        if piece in black:
            black_scores += black[piece]
        else:
            white_scores += white[piece]

    return {'black': black_scores, 'white': white_scores}
