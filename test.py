import numpy as np
import cv2
import os
from matplotlib import pyplot as plt


img = cv2.imread('../bad/1.png')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

for BlockSize in range(11, 501, 10):
    thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, BlockSize, 0)
    cv2.imshow('bin ' + str(BlockSize), thresh)
    cv2.waitKey(0)

for C in range(-50 , 50, 5):
    thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 501, C)
    cv2.imshow('bin', thresh)
    cv2.waitKey(0)