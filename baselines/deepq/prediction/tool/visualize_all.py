import cv2
import numpy as np
import sys

ss = np.load(sys.argv[1]).astype(np.uint8)
for s in ss
    for i in range(0, 12, 3):
        cv2.imshow('img%d' % i,s[:,:,i:i+3])
    cv2.waitKey(0)
