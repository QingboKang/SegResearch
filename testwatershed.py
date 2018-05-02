import numpy as np
import cv2
import os
from matplotlib import pyplot as plt
import normalizeStaining
import evaluation
from PIL import Image


def ImageWatershed(srcImageName, dstImageName, dstMarkName):
    img = cv2.imread(srcImageName)

    '''Color Normalization'''
    img = img.astype('float32')
    img = normalizeStaining.normalizeStaining(img)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    ddepth = cv2.CV_32F
    dx = cv2.Sobel(gray, ddepth, 1, 0)
    dy = cv2.Sobel(gray, ddepth, 0, 1)
    dxabs = cv2.convertScaleAbs(dx)
    dyabs = cv2.convertScaleAbs(dy)
    thresh1 = cv2.addWeighted(dxabs, 0.5, dyabs, 0.5, 0)
    #ret, thresh = cv2.threshold(gray,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
    thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 501, 31)

    # noise removal
    kernel = np.ones((3, 3), np.uint8)
    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=3)

    # sure background area
    sure_bg = cv2.dilate(opening, kernel, iterations=3)

    # Finding sure foreground area
    dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 3)
  #  cv2.imshow('dist_transform', dist_transform);

    ret, sure_fg = cv2.threshold(dist_transform, 0.28*dist_transform.max(), 255, 0)
   # cv2.imshow('foreground', sure_fg);
    #cv2.imwrite('fg.png', sure_fg);

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
   # bin_mask[markers == -1] = 255
    bin_mask[markers == 1] = 0
#    bin_mask[bin_mask == -1] = 0;
    bin_mask = bin_mask.astype(np.uint8)
    #print (bin_mask.shape )

   # cv2.imshow('test', img);
   # cv2.imshow('bin_mask', bin_mask);
    cv2.imwrite( dstMarkName, bin_mask);

    ratio = evaluation.get_white_ratio(bin_mask)
    area_count, area_mean, area_std = evaluation.count_contours_stddev(bin_mask)

  #  cv2.waitKey(0);
 #   cv2.imwrite(dstImageName, img);
    cv2.destroyAllWindows();

    return ratio, area_count, area_mean, area_std;



def ImageWatershed_1(srcImageName, dstImageName, dstMarkName):
    img = cv2.imread(srcImageName)

    '''Color Normalization'''
    img = img.astype('float32')
    img = normalizeStaining.normalizeStaining(img)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    ddepth = cv2.CV_32F
    dx = cv2.Sobel(gray, ddepth, 1, 0)
    dy = cv2.Sobel(gray, ddepth, 0, 1)
    dxabs = cv2.convertScaleAbs(dx)
    dyabs = cv2.convertScaleAbs(dy)
    thresh1 = cv2.addWeighted(dxabs, 0.5, dyabs, 0.5, 0)
    #ret, thresh = cv2.threshold(gray,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
    thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 501, 0)

    # noise removal
    kernel = np.ones((3, 3), np.uint8)
    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=3)

    # sure background area
    sure_bg = cv2.dilate(opening, kernel, iterations=3)

    # Finding sure foreground area
    dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 3)
  #  cv2.imshow('dist_transform', dist_transform);

    ret, sure_fg = cv2.threshold(dist_transform, 0.28*dist_transform.max(), 255, 0)
   # cv2.imshow('foreground', sure_fg);
    #cv2.imwrite('fg.png', sure_fg);

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
   # bin_mask[markers == -1] = 255
    bin_mask[markers == 1] = 0
#    bin_mask[bin_mask == -1] = 0;
    bin_mask = bin_mask.astype(np.uint8)
    #print (bin_mask.shape )

   # cv2.imshow('test', img);
   # cv2.imshow('bin_mask', bin_mask);
    cv2.imwrite( dstMarkName, bin_mask);

    ratio = evaluation.get_white_ratio(bin_mask)
    area_count, area_mean, area_std = evaluation.count_contours_stddev(bin_mask)

  #  cv2.waitKey(0);
 #   cv2.imwrite(dstImageName, img);
    cv2.destroyAllWindows();

    return ratio, area_count, area_mean, area_std;


#tif_dir = '../Tissue images/'

tif_dir = '../bad/'
bad_dir = '../bad/'
#result_dir = 'results/'
result_dir = '../water_1/'

tif_files = os.listdir(tif_dir)
count = 1
for file in tif_files:
    if not os.path.isdir(file):
        tiffilename = tif_dir + file
        resultFileName = result_dir + file[0:file.find('.tif')] + '.png'
        maskFileName = result_dir + file[0:file.find('.tif')] + '.png'

        ratio, area_count, area_mean, area_std = ImageWatershed_1(tiffilename, resultFileName, maskFileName)

        count += 1
        print(maskFileName, " ", str(ratio), " ", str(area_count), " ", str(area_mean), " ", str(area_std))
        '''
        count_holes, std_holes = ImageWatershed1(tiffilename, resultFileName, maskFileName)
        count += 1
        '''




