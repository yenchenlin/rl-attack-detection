import cv2
import numpy as np
import sys

img = np.load(sys.argv[1]).astype(np.uint8)
cv2.imshow('img', img)
cv2.waitKey(0)
