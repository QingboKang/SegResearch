import numpy as np
import cv2
import os
import random

def randomExtractPatches( image_index ):
    orig_image_dir = "../../Tissue images pad/"
    watershed_image_dir = "../../water pad/result_"
    mask_image_dir = "../../Mask/"

    # save dir
    patch_savedir = "../../PatchImages_test/"

    # file name
    orig_image_name = orig_image_dir + str(image_index) + ".png"
    watershed_image_name = watershed_image_dir + str(image_index) + ".png"
    mask_image_name = mask_image_dir + str(image_index) + ".png"
    # images
    orig_image = cv2.imread(orig_image_name)
    watershed_image = cv2.imread(watershed_image_name)
    mask_image = cv2.imread(mask_image_name)
    #
    mask_image_pad = np.pad(mask_image, ((25, 25), (25, 25), (0, 0)), 'constant', constant_values=0)
    #
    total_list = []
    count_0 = 0
    count_1 = 0
    count_patches = 0
    for i in range(0, 100000):
        if count_0 >= 500 and count_1 >= 500:
            break
       # print (str(count_0) + ":" + str(count_1) + "  " + str(len(total_list)))
        [x, y] = [random.randint(25, 1024), random.randint(25, 1024)]

        if [x, y] not in total_list:
            total_list.append([x,y])

            # corresponding label
            pix_value = mask_image[y - 25, x - 25]
            if np.all(pix_value == 0):
                label = 0
                count_0 += 1
                if count_0 > 500:
                    continue;
            else:
                label = 1
                count_1 += 1
                if count_1 > 500:
                    continue

            start_x = x - 25
            end_x = x + 25 + 1
            start_y = y - 25
            end_y = y + 25 + 1

            # patch image
            patch_orig_image = orig_image[start_y: end_y, start_x: end_x, :]
            patch_ws_image = watershed_image[start_y: end_y, start_x: end_x]
    #       patch_mask_image = mask_image_pad[start_y: end_y, start_x: end_x]

            # save file names
            patch_origsavename = patch_savedir + "orig_" + str(image_index) + "_" + str(count_patches) + "_" + str(label) + ".png"
            patch_wssavename = patch_savedir + "watershed_" + str(image_index) + "_" + str(count_patches) + "_" + str(label) + ".png"
   #        patch_masksavename = patch_savedir + "mask_" + str(image_index) + "_" + str(count_patches) + "_" + str(label) + ".png"

            cv2.imwrite(patch_origsavename, patch_orig_image)
            cv2.imwrite(patch_wssavename, patch_ws_image)
   #        cv2.imwrite(patch_masksavename, patch_mask_image)

            count_patches += 1


for i in range(1, 31):
    randomExtractPatches(i)
    print(i)