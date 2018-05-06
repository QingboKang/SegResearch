import numpy as np
import cv2
import os

def PadImage( orig_image ):
    w = 51  # window size
    p = (w - 1) / 2  # padding

    p = int(p)
    image = np.pad(orig_image, ((p, p), (p, p), (0, 0)), 'constant', constant_values=0)
    return image

img_dir = "../../water/"
save_dir = "../../water pad/"

tif_files = os.listdir(img_dir)
for file in tif_files:
    if not os.path.isdir(file):
        tiffilename = img_dir + file
        savefilename = save_dir + file[0:file.find('.png')] + '.png'

        orig_image = cv2.imread(tiffilename)
        [height, width, channels] = orig_image.shape
        orig_image[0, : ] = 0
        orig_image[height - 1, :] = 0
        orig_image[ : , 0] = 0
        orig_image[ : , width - 1] = 0
        image = PadImage(orig_image)
        cv2.imwrite( savefilename, image)



