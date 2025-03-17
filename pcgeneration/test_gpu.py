import cv2
import numpy as np
import time
import open3d as o3d
import warnings
warnings.filterwarnings('ignore')
from cvfun import *
import os
stereo_parematers = read_stereo_paremater('/home/airlab/Desktop/Thinh/pcgeneration/stereoMap_50.txt')
Q = stereo_parematers[4]
# Create a StereoSGBM object with desired parameters
min_disparity = 72#nsize 25: 72, 50: 128; 512= 32
# num_disparities = 16 * 5  # Must be divisible by 16
num_disparities = 512 # Must be divisible by 16  # size 25: 256, 50: 128
uniqueness_ratio = 15
folder_path = "/home/airlab/Desktop/Thinh/pcgeneration/SIZE_50"
num_pairs = 58
size = 50
stereo = cv2.cuda.createStereoSGM(min_disparity, num_disparities)
totaltime = []
if __name__ =='__main__':
    for i in range(0, num_pairs + 1):
        # left_image_filename = f'L_0_{size}.png'
        # right_image_filename = f'R_0_{size}.png' 
        # mask_img_filename = f'mask_L_0_{size}.png'
        left_image_filename = f'L_{i}_{size}.png'
        right_image_filename = f'R_{i}_{size}.png' 
        mask_img_filename = f'mask_L_{i}_{size}.png'
        # left_image_filename = 'mask_color_L.png'
        # right_image_filename = 'mask_color_R.png'
        # mask_img_filename = 'mask_L.png'
        # Construct full paths to the images
        left_image_path = os.path.join(folder_path, left_image_filename)
        right_image_path = os.path.join(folder_path, right_image_filename)
        mask_image_path = os.path.join(folder_path, mask_img_filename)
        left_image = read_img(left_image_path, False)
        right_image = read_img(right_image_path , False)
        mask_L = read_img(mask_image_path, False)
        left_image_color = read_img(left_image_path, True)  
        h,w = left_image_color.shape[0:2]
        t1 = time.time()
        # Check if images are loaded correctly
        if left_image is None or right_image is None:
            raise ValueError("Could not open or find the images!")
        # Define padding size
        padding_ = 7# padding 256 = 50
        # Get the dimensions of the original image
        height_, width_ = left_image.shape
        agg_l, agg_r = split_and_pad_images(left_image, right_image, 3, padding_)
        merge_dis = []
        for im_l, im_r in zip(agg_l, agg_r):
            left_gpu = cv2.cuda_GpuMat()
            right_gpu = cv2.cuda_GpuMat()
            left_gpu.upload(im_l)
            right_gpu.upload(im_r)
            # stereo = cv2.cuda.createStereoSGM(min_disparity, num_disparities, sc_P1,sc_P2,uniqueness_ratio)
            # stereo = cv2.cuda.createStereoSGM(min_disparity, num_disparities)
            disparity_gpu = stereo.compute(left_gpu, right_gpu)
            disparity_map = disparity_gpu.download()
            merge_dis.append(disparity_map)
        final_dis = assemble_disparity_map(merge_dis, height_, width_, 3, padding_)
        # print("min dis:", np.min(final_dis))
        # print("max dis:", np.max(final_dis))
        # filtered_disp_vis = final_dis.astype(np.float32) / 16
        filtered_disp_vis = specklefilter_dis(final_dis)

        depth = cv2.bitwise_and(filtered_disp_vis, filtered_disp_vis, mask = mask_L)
        Q1 = np.float32([[1, 0, 0, -w / 2.0],
                        [0, -1, 0, h / 2.0],
                        [0, 0, 0, -Q[2, 3]],
                        [0, 0, Q[3, 2], Q[3, 3]]])
        pcPoints = cv2.reprojectImageTo3D(depth, Q1)
        pcColors = cv2.cvtColor(left_image_color, cv2.COLOR_BGR2RGB)
        mask_ = depth > depth.min()  + 72
        pcColors = pcColors[mask_]
        pcPoints = pcPoints[mask_]
        print("Total time", time.time()-t1)
        t2= time.time()
        e = t2-t1
        totaltime.append(e)
        show(pcPoints, pcColors)
    
    average = sum(totaltime) / len(totaltime)
    print("Average:", average)
   
# ---------------------

    # for i in range(0, num_pairs + 1):
    #     # left_image_filename = f'L_0_{size}.png'
    #     # right_image_filename = f'R_0_{size}.png' 
    #     # mask_img_filename = f'mask_L_0_{size}.png'
    #     left_image_filename = f'L_{i}_{size}_mask.png'
    #     right_image_filename = f'R_{i}_{size}_mask.png' 
    #     mask_img_filename = f'mask_L_{i}_{size}.png'
    #     left_image_path = os.path.join(folder_path, left_image_filename)
    #     right_image_path = os.path.join(folder_path, right_image_filename)
    #     mask_image_path = os.path.join(folder_path, mask_img_filename)
    #     left_image = read_img(left_image_path, False)
    #     right_image = read_img(right_image_path , False)
    #     mask_L = read_img(mask_image_path, False)
    #     left_image_color = read_img(left_image_path, True)  
    #     h,w = left_image_color.shape[0:2]
    #     t1 = time.time()
    #     # Upload images to GPU
    #     left_gpu = cv2.cuda_GpuMat()
    #     right_gpu = cv2.cuda_GpuMat()
    #     left_gpu.upload(left_image)
    #     right_gpu.upload(right_image)
    #     # stereo = cv2.cuda.createStereoSGM(min_disparity, num_disparities, sc_P1,sc_P2,uniqueness_ratio)
    #     stereo = cv2.cuda.createStereoSGM(min_disparity, num_disparities)
    #     # Compute the disparity map
    #     disparity_gpu = stereo.compute(left_gpu, right_gpu)
    #     t2 = time.time()
    #     # right_gpu.release()
    #     # left_gpu.release()
    #     # stereo.release()
    #     # Download the disparity map back to CPU
    #     disparity_map = disparity_gpu.download()
    #     # Normalize the disparity map for visualization
    #     # disparity = cv2.normalize(disparity_map, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    #     # disparity = disparity_map.astype(np.float32) / 16
    #     disparity = specklefilter_dis(disparity_map)
    #     depth = cv2.bitwise_and(disparity, disparity, mask = mask_L)
    #     Q1 = np.float32([[1, 0, 0, -w / 2.0],
    #                     [0, -1, 0, h / 2.0],
    #                     [0, 0, 0, -Q[2, 3]],
    #                     [0, 0, Q[3, 2], Q[3, 3]]])
    #     pcPoints = cv2.reprojectImageTo3D(depth, Q1)
    #     pcColors = cv2.cvtColor(left_image_color, cv2.COLOR_BGR2RGB)
    #     mask_ = depth > depth.min()  + 72
    #     pcColors = pcColors[mask_]
    #     pcPoints = pcPoints[mask_] 
    #     print("Total time", t2-t1)
    #     e = t2-t1
    #     totaltime.append(e)
    #     # show(pcPoints, pcColors)
    # average = sum(totaltime) / len(totaltime)
    # print("Average:", average) 
