import numpy as np
import cv2
import os
from matplotlib import pyplot as plt
import evaluation
import normalizeStaining


def ImageWatershed(img, isNormalize, dstImageName, dstMarkName, blocksize, constant):
    '''Color Normalization'''
    if isNormalize == True:
        img = img.astype('float32')
        img = normalizeStaining.normalizeStaining(img)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, blocksize, constant)

    # noise removal
    kernel = np.ones((3, 3), np.uint8)
    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=3)

    # sure background area
    sure_bg = cv2.dilate(opening, kernel, iterations=3)

    # Finding sure foreground area
    dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 3)
    ret, sure_fg = cv2.threshold(dist_transform, 0.28*dist_transform.max(), 255, 0)

    # Finding unknown region
    sure_fg = np.uint8(sure_fg)
    unknown = cv2.subtract(sure_bg, sure_fg)

    # Marker labelling
    ret, markers = cv2.connectedComponents(sure_fg)

    # Add one to all labels so that sure background is not 0, but 1
    markers = markers+1

    # Now, mark the region of unknown with zero
    markers[unknown == 255] = 0

    markers = cv2.watershed(img, markers)
    img[markers == -1] = [255, 0, 0]

    bin_mask = np.ones(markers.shape) * 255
    bin_mask[markers == 1] = 0
    bin_mask = bin_mask.astype(np.uint8)

    ratio = evaluation.get_white_ratio(bin_mask)
    area_count, area_mean, area_std = evaluation.count_contours_stddev(bin_mask)

  #  print (str(constant), ": ", str(ratio), " ", str(area_count), " ", str(area_mean), " ", str(area_std))

    #cv2.imshow('binary', thresh);
    return bin_mask, ratio, area_count,  area_mean, area_std


# tif_dir = '../Tissue images/'
tif_dir = "../Tissue images/"

result_dir = '../water/result_'


const = range(-20, 51, 5)
blocksizes = range(251, 653, 100)
listIsNorm = [False, True]

listRatio = []
listCount = []

tif_files = os.listdir(tif_dir)
count = 0
maxNumberOfContours = 0
for file in tif_files:
    if not os.path.isdir(file):
        maxNumberOfContours = 0
        tiffilename = tif_dir + file
        resultFileName = result_dir + file[0:file.find('.tif')] + '.png'

        for size in blocksizes:
            for c in const:
                for bNorm in listIsNorm:
                    img = cv2.imread(tiffilename)
                    bin_mask, ratio, area_count, area_mean, area_std = ImageWatershed(img, bNorm, "", "", size, c)

                    if ratio >= 0.15 and ratio < 0.4 and maxNumberOfContours < area_count:
                     #   resultFileName = result_dir + file[0:file.find('.tif')] + "_" + str(count) + '.png'
                        maxNumberOfContours = area_count
                        cv2.imwrite(resultFileName, bin_mask)
                        print(resultFileName, " ", str(ratio), " ", str(area_count), " ", str(bNorm), " ", str(size), " ", str(c))
                        count += 1


