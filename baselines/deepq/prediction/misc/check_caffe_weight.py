import numpy as np
import cPickle as pickle
import sys

data = pickle.load(open(sys.argv[1], "rb" ))
for key in data:
    print key, data[key].shape
