import numpy as np
import os, sys, cv2
import glob
from tqdm import *

from episode_reader import EpisodeReader

if __name__ == '__main__':
    path = sys.argv[1]
    mean_path = sys.argv[2]

    mean = np.zeros([84, 84, 1], dtype=np.float64)
    n = 0
    for path in tqdm(glob.glob(os.path.join(path, '*.tfrecords'))):
        try:
            reader = EpisodeReader(path)
            for s, a, x in reader.read():
                mean += s[:,:,-1:]
                n += 1
        except:
            print ('Fail to load %s' % path)
    mean /= n
    np.save(mean_path, mean)
