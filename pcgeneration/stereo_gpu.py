import cv2
import numpy as np
import time
import open3d as o3d
import warnings
warnings.filterwarnings('ignore')
from cvfun import *
import os
stereo_parematers = read_stereo_paremater('/home/airlab/Desktop/Thinh/pcgeneration/stereoMap_25.txt')
Q = stereo_parematers[4]
# Create a StereoSGBM object with desired parameters
min_disparity = 72 #nsize 25: 105, 50: 150
# num_disparities = 16 * 5  # Must be divisible by 16
num_disparities = 256 # Must be divisible by 16  # size 25: 256, 50: 128
# max_disparity = 256
block_size = 11
uniqueness_ratio = 10
# sc_P1 = 8*block_size**2
sc_P1 = 8*block_size**2
# print("P1", sc_P1)
# sc_P2 = 32*block_size**2
sc_P2 = 32*block_size**2
# sc_P1  = 50
# sc_P2 = 100
# t1 = time.time()
# print("Time 1", time.time()- t1)
folder_path = "/home/airlab/Desktop/Thinh/pcgeneration/SIZE_25"
num_pairs = 58
size = 25

totaltime = []
if __name__ =='__main__':

    for i in range(0, num_pairs + 1):
        left_image_filename = f'L_{i}_{size}.png'
        right_image_filename = f'R_{i}_{size}.png' 
        mask_img_filename = f'mask_L_{i}_{size}.png'
        # Construct full paths to the images
        left_image_path = os.path.join(folder_path, left_image_filename)
        right_image_path = os.path.join(folder_path, right_image_filename)
        mask_image_path = os.path.join(folder_path, mask_img_filename)
        left_image = read_img(left_image_path, False)
        right_image = read_img(right_image_path , False)
        mask_L = read_img(mask_image_path, False)
        left_image_color = read_img(left_image_path, True)  
        h,w = left_image_color.shape[0:2]
        # Check if images are loaded correctly
        if left_image is None or right_image is None:
            raise ValueError("Could not open or find the images!")
        t1 = time.time()
        
        # Upload images to GPU
        left_gpu = cv2.cuda_GpuMat()
        right_gpu = cv2.cuda_GpuMat()
        left_gpu.upload(left_image)
        right_gpu.upload(right_image)
        # stereo = cv2.cuda.createStereoSGM(min_disparity, num_disparities, sc_P1,sc_P2,uniqueness_ratio)
        stereo = cv2.cuda.createStereoSGM(min_disparity, num_disparities)
        # Compute the disparity map
        disparity_gpu = stereo.compute(left_gpu, right_gpu)
        # right_gpu.release()
        # left_gpu.release()
        # stereo.release()
        # Download the disparity map back to CPU
        disparity_map = disparity_gpu.download()
        # Normalize the disparity map for visualization
        # disparity = cv2.normalize(disparity_map, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        disparity = disparity_map.astype(np.float32) / 16
        depth = cv2.bitwise_and(disparity, disparity, mask = mask_L)
        Q1 = np.float32([[1, 0, 0, -w / 2.0],
                        [0, -1, 0, h / 2.0],
                        [0, 0, 0, -Q[2, 3]],
                        [0, 0, Q[3, 2], Q[3, 3]]])
        pcPoints = cv2.reprojectImageTo3D(depth, Q1)
        pcColors = cv2.cvtColor(left_image_color, cv2.COLOR_BGR2RGB)
        mask_ = depth > depth.min()  + 105
        pcColors = pcColors[mask_]
        pcPoints = pcPoints[mask_]
        t2 = time.time()
        # print("Total time", t2-t1)
        show(pcPoints, pcColors)
        e = t2-t1
        totaltime.append(e)
    average = sum(totaltime) / len(totaltime)
    print("Average:", average)
