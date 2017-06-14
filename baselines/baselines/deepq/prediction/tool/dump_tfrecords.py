import numpy as np
from episode_reader import EpisodeReader

import sys, os

if __name__ == '__main__':
    reader = EpisodeReader(path=sys.argv[1], height=84, width=84)
    dir = sys.argv[2]
    i = 0
    for s, a, x_t_1 in reader.read():
        np.save(os.path.join(dir, '%05d-s') % i, s)
        np.save(os.path.join(dir, '%05d-x_t_1') % i, x_t_1)
        np.save(os.path.join(dir, '%05d-a' % i), np.asarray([a]))
        i += 1
