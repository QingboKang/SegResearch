import numpy as np
import cv2

def CountContoursStddev(binImg):
    colorImg = cv2.cvtColor(binImg, cv2.COLOR_GRAY2BGR)
    im2, cnts, hierchy = cv2.findContours( binImg.copy(), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE )

    lstAreas = []
    area_count = 0;
    for c in cnts:
        area = cv2.contourArea(c)
        if area < binImg.size * 0.2:
            area_count += 1
            lstAreas.append(area)
            color = list(np.random.random(size=3) * 256)
            cv2.drawContours(colorImg, [c], -1, color, 2)

    arr = np.array(lstAreas)
    area_std = np.std(arr)
    area_mean = np.mean(arr)

    cv2.imshow('cont', colorImg)
    cv2.waitKey(0)
    return area_count, area_std, area_mean


imgPath = "../Mask/29.png"
img = cv2.imread(imgPath)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
areas, std, mean = CountContoursStddev(gray)
print (areas, " ", std, "  ", mean)
