import numpy as np
import cv2
import os

def patch(orig_image, ws_image, image_index, mask_image, orig_patches_savedir, ws_patches_savedir):
    # [1050, 1050]
    [height, width, channels] = orig_image.shape

    w = 51  # window size
    p = (w - 1) / 2  # padding
    p = int(p)
    count_patches = 0
    for i in range(p, height - p, 1):
        for j in range(p, width - p, 1):
            start_x = j - p
            end_x = j + p + 1
            start_y = i - p
            end_y = i + p + 1

            # patch image
            patch_orig_image = orig_image[start_y: end_y, start_x: end_x]
            patch_ws_image = ws_image[start_y: end_y, start_x: end_x]

            # corresponding label
            pix_value = mask_image[i, j]
            if np.all(pix_value == 0):
                label = 0
            else:
                label = 1

            # save names
            patch_orig_savename = orig_patches_savedir +  "original_" + str(image_index) + "_" + str(count_patches) + "_" + str(label) + ".png"
            patch_ws_savename = ws_patches_savedir + "water_" + str(image_index) + "_" + str(count_patches) + "_" + str(label) + ".png"

        #    print (patch_orig_savename)
        #    print (patch_ws_savename)
            cv2.imwrite(patch_orig_savename, patch_orig_image)
            cv2.imwrite(patch_ws_savename, patch_ws_image)

            count_patches += 1


orig_image_dir = "../../Tissue images pad/"
watershed_image_dir = "../../water pad/result_"
mask_image_dir = "../../Mask/"

# save dir
orig_patches_savedir = "../../PatchImages/original/"
ws_patches_savedir = "../../PatchImages/watershed/"

orig_image_files = os.listdir(orig_image_dir)
for file in orig_image_files:
    if not os.path.isdir(file):
        orig_imagename = orig_image_dir + file

        # image index
        index = int(file[0:file.find('.png')])

        watershed_imagename = watershed_image_dir + str(index) + ".pn.png"
        mask_imagename = mask_image_dir + str(index) + ".png"

        # read images
        orig_image = cv2.imread(orig_imagename)
        watershed_image = cv2.imread(watershed_imagename)
        mask_image = cv2.imread(mask_imagename)

        patch(orig_image, watershed_image, index, mask_image, orig_patches_savedir, ws_patches_savedir)



