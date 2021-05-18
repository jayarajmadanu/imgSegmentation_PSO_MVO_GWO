import cv2
import numpy as np
from matplotlib import pyplot as plt
from cost_functions import getFunction
from GWO import GWO
import time


def hist(img):
    w, h = img.shape
    H = np.zeros(256)
    for i in range(w):
        for j in range(h):
            H[img[i, j] ] = H[img[i, j] ] + 1
    H = H/sum(H)
    return H


img = cv2.imread('../../images/horse.png', 0)
img = cv2.resize(img, (220,220))
cv2.imshow('Input image', img)
cv2.waitKey()

H = hist(img)

SearchAgents_no = 50
Function_name = 'F0'
Max_iteration = 500

list = getFunction(Function_name)
fobj = list[0]
lb = list[1]
ub = list[2]
dim = list[3]

start = time.time()
result = GWO(SearchAgents_no, Max_iteration, lb, ub, dim, fobj, H)
print('seconds took to complete ', time.time()-start)

best_score = result[0]
best_position = result[1]
print('Threshold ', best_score)
width, height = img.shape

for i in range(width):
    for j in range(height):
        if img[i, j] < best_score:
            img[i, j] = 0
        else:
            img[i, j] = 255

cv2.imshow('segmented image', img)
cv2.waitKey()
cv2.imwrite('../../images/gwoOutput.jpg', img)
