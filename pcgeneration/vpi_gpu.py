
import sys
import vpi
import numpy as np
from PIL import Image
from argparse import ArgumentParser
import cv2
import time
from cvfun import *
import os
stereo_parematers = read_stereo_paremater('/home/airlab/Desktop/Thinh/stereoMap_50.txt')
Q = stereo_parematers[4]
scale = 1 # pixel value scaling factor when loading input
backend = vpi.Backend.CUDA
folder_path = "/home/airlab/Desktop/Thinh/SIZE_50"
num_pairs = 58
size = 50
if __name__ =='__main__':

    for i in range(0, num_pairs + 1):
        left_image_filename = f'L_{i}_{size}.png'
        right_image_filename = f'R_{i}_{size}.png' 
        mask_img_filename = f'mask_L_{i}_{size}.png'
        # Construct full paths to the images
        left_image_path = os.path.join(folder_path, left_image_filename)
        right_image_path = os.path.join(folder_path, right_image_filename)
        mask_image_path = os.path.join(folder_path, mask_img_filename)
        # left_image = read_img(left_image_path, False)
        # right_image = read_img(right_image_path , False)
        mask_L = read_img(mask_image_path, False)
        left_image_color = read_img(left_image_path, True)  
        downscale =  1
        window_size = 7
        conf_threshold = 32767
        p1 = 100
        p2 = 150
        p2_alpha = 2
        uniqueness = 10
        num_passes = 3
        min_disparity = 0
        max_disparity = 256
        conf_type = 'best'
        skip_diagonal = False
        skip_confidence = False
        if conf_type == 'best':
            conftype = vpi.ConfidenceType.INFERENCE if backend == 'ofa-pva-vic' else vpi.ConfidenceType.ABSOLUTE
            # # conftype = vpi.ConfidenceType.INFERENCE
            # conftype = vpi.ConfidenceType.RELATIVE
        else:
            raise ValueError(f'E Invalid confidence type: {conf_type}')
        minDisparity = min_disparity
        maxDisparity = max_disparity
        includeDiagonals = not skip_diagonal
        numPasses = num_passes
        calcConf = not skip_confidence
        downscale = 1
        windowSize = window_size
        quality = 6
        t1 = time.time()
        # mask_L = cv2.imread('/home/airlab/Desktop/Thinh/SIZE_25/mask_L_25.png',0)
        pil_left = Image.open(left_image_path)
        np_left = np.asarray(pil_left)
        # pil_left = read_raw_file(left_img, resize_to=[-1, -1], verbose=False)
        # np_left = np.asarray(pil_left)
        # pil_right = read_raw_file(right_img, resize_to=[-1, -1], verbose=False)
        # np_right = np.asarray(pil_right)
        pil_right = Image.open(right_image_path)
        np_right = np.asarray(pil_right)
        # Streams for left and right independent pre-processing
        streamLeft = vpi.Stream()
        streamRight = vpi.Stream()
        with vpi.Backend.CUDA:
            with streamLeft:
                left = vpi.asimage(np_left).convert(vpi.Format.Y16_ER, scale=scale)
            with streamRight:
                right = vpi.asimage(np_right).convert(vpi.Format.Y16_ER, scale=scale)
        confidenceU16 = None
        outWidth = (left.size[0] + downscale - 1) // downscale
        outHeight = (left.size[1] + downscale - 1) // downscale
        confidenceU16 = vpi.Image((outWidth, outHeight), vpi.Format.U16)
        # Use stream left to consolidate actual stereo processing
        streamStereo = streamLeft
        # Estimate stereo disparity.
        with streamStereo, backend:
            disparityS16 = vpi.stereodisp(left, right, downscale=downscale, out_confmap=confidenceU16,
                                        window=windowSize, maxdisp=maxDisparity, confthreshold=conf_threshold,
                                        quality=quality, conftype=conftype, mindisp=minDisparity,
                                        p1=p1, p2=p2, p2alpha=p2_alpha, uniqueness=uniqueness,
                                        includediagonals=includeDiagonals, numpasses=numPasses)
        # Postprocess results and save them to disk
        with streamStereo, vpi.Backend.CUDA:
            # Some backends outputs disparities in block-linear format, we must convert them to
            # pitch-linear for consistency with other backends.
            if disparityS16.format == vpi.Format.S16_BL:
                disparityS16 = disparityS16.convert(vpi.Format.S16, backend=vpi.Backend.VIC)
            # Scale disparity and confidence map so that values like between 0 and 255.
            # Disparities are in Q10.5 format, so to map it to float, it gets
            # divided by 32. Then the resulting disparity range, from 0 to
            # stereo.maxDisparity gets mapped to 0-255 for proper output.
            # Copy disparity values back to the CPU.
            disparityU8 = disparityS16.convert(vpi.Format.U8, scale=255.0/(32*maxDisparity)).cpu()
        disparity = disparityU8.astype(np.float32) / 16

        disparity = cv2.bitwise_and(disparity, disparity, mask = mask_L)
        # depth = np.where(mask_L > 1e-5, disparity, 0)
        left_image_color = cv2.imread(left_image_path)
        h,w = left_image_color.shape[0:2]
        #Q matrix is form during the calibration
        Q1 = np.float32([[1, 0, 0, -w / 2.0],
                        [0, -1, 0, h / 2.0],
                        [0, 0, 0, -Q[2, 3]],
                        [0, 0, Q[3, 2], Q[3, 3]]])
        pcPoints = cv2.reprojectImageTo3D(disparity, Q1)
        pcColors = cv2.cvtColor(left_image_color, cv2.COLOR_BGR2RGB)
        mask_ = disparity > disparity.min() + 8
        pcColors = pcColors[mask_]
        pcPoints = pcPoints[mask_]
        print("Total time:", time.time() - t1)
        # print(pcPoints)
        show(pcPoints, pcColors)