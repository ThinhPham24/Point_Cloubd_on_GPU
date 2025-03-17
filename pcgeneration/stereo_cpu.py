import cv2
import time
import numpy as np
import open3d as o3d
from cvfun import *
import os

stereo_parematers = read_stereo_paremater('/home/airlab/Desktop/Thinh/pcgeneration/stereoMap_50.txt')
Q = stereo_parematers[4]

# # Load left and right images in grayscale
# left_image = read_img("/home/airlab/Desktop/Thinh/SIZE_50/L_0_50.png", False)
# right_image = read_img("/home/airlab/Desktop/Thinh/SIZE_50/R_0_50.png", False)
# mask_L = read_img("/home/airlab/Desktop/Thinh/SIZE_50/mask_L_0_50.png", False)
# left_image_color = read_img("/home/airlab/Desktop/Thinh/SIZE_50/L_0_50.png", True)
'''
Size 25: 
window_size = 7
min_disp = 50
num_disp = 256
P1_s = 100
P2_s= 150
Size 50:
Size 75:
'''
folder_path = "/home/airlab/Desktop/Thinh/pcgeneration/SIZE_50"
num_pairs = 58
size = 50
totaltime = []
window_size = 7
min_disp = 128 # 128
# max_disp = min_disp * 9
# num_disp = max_disp - min_disp   # Needs to be divisible by 16
num_disp = 256
stereo = cv2.StereoSGBM_create(minDisparity = min_disp,
    numDisparities = num_disp,
    blockSize = 7,
    P1 =   8*3*window_size**2,
    P2 =  32*3*window_size**2,
    disp12MaxDiff = 1,
    uniquenessRatio = 12,
    speckleWindowSize = 100,
    speckleRange = 32
)
if __name__ =='__main__':

    for i in range(0, num_pairs + 1):
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
        mask_L = read_img(left_image_path, False)
        left_image_color = read_img(left_image_path, True) 
        t1 = time.time() 
        # window_size = 7
        # min_disp = 128 # 128
        # # max_disp = min_disp * 9
        # # num_disp = max_disp - min_disp   # Needs to be divisible by 16
        # num_disp = 256
        # t1 = time.time()
        # stereo = cv2.StereoSGBM_create(minDisparity = min_disp,
        #     numDisparities = num_disp,
        #     blockSize = 7,
        #     P1 =   8*3*window_size**2,
        #     P2 =  32*3*window_size**2,
        #     disp12MaxDiff = 1,
        #     uniquenessRatio = 12,
        #     speckleWindowSize = 100,
        #     speckleRange = 32
        # )
        # print("Time 2", time.time() - t1)
        # print('computing disparity...')
        # disp = stereo.compute(imgL, imgR).astype(np.float32) / 16.0
        # stereo =  cv2.StereoSGBM_create(numDisparities=num_disp, minDisparity= min_disp,P1 =8*3*window_size**2,P2= 32*3*window_size**2,  blockSize=7, uniquenessRatio = 12, speckleWindowSize = 100,speckleRange = 32)
        disparity = stereo.compute(left_image, right_image)
        # disparity = cv2.normalize(disparity, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        disparity = disparity.astype(np.float32) / 16
        depth = cv2.bitwise_and(disparity, disparity, mask = mask_L)
        h,w = left_image.shape[0:2]
        # print("Dis time", time.time()-t1)
        # t2 = time.time()
        Q1 = np.float32([[1, 0, 0, -w / 2.0],
                        [0, -1, 0, h / 2.0],
                        [0, 0, 0, -Q[2, 3]],
                        [0, 0, Q[3, 2], Q[3, 3]]])
        pcPoints = cv2.reprojectImageTo3D(depth, Q1)
        pcColors = cv2.cvtColor(left_image_color, cv2.COLOR_BGR2RGB)
        mask_ = depth > depth.min()  
        pcColors = pcColors[mask_]
        pcPoints = pcPoints[mask_]
        # left_gpu.release()
        # right_gpu.release()
        t3 = time.time()
        e = t3-t1
        totaltime.append(e)
        print("Total time", t3-t1)
        # print("Point cloud:", pcPoints)   
        show(pcPoints, pcColors)
        # if cv2.waitKey(0) & 0xFF == ord('q'):  # Press 'q' to quit early
        #     break
    average = sum(totaltime) / len(totaltime)
    print("Average:", average)