import cv2
import numpy as np
import time
import os
import glob
from cvfun import resizeImage, read_stereo_paremater, remap_img
# Check if images are loaded correctly
folder_path = '/home/airlab/Desktop/Thinh/pcgeneration/Data_images'
out_folder = '/home/airlab/Desktop/Thinh/pcgeneration/SIZE_25/'

stereo_parematers = read_stereo_paremater('/home/airlab/Desktop/Thinh/pcgeneration/stereoMap_25.txt')
R_x = stereo_parematers[0]
R_y = stereo_parematers[1]
L_x = stereo_parematers[2]
L_y = stereo_parematers[3]

file_path = glob.glob(os.path.join(folder_path, 'R_*.png'))
size = 25
map = [R_x, R_y]
# print('map:',map[0])
def nothing(x):
        pass
for i, image_path in enumerate(file_path):
    print("image path:", image_path)
    file_name = os.path.basename(image_path)       # e.g., 'image.png'
    name, ext = os.path.splitext(file_name)
    
    img = cv2.imread(image_path)
    re_image = resizeImage(img, size)
    img_map = remap_img(re_image,map[0], map[1])
    
    image_hsv = cv2.cvtColor(img_map, cv2.COLOR_BGR2HSV)
    # image_hsv = resizeImage(image_hsv,50)
    '''
    cv2.namedWindow("Trackbars")

    # Create trackbars for lower and upper HSV bounds
    cv2.createTrackbar("Lower H", "Trackbars", 50, 179, nothing)
    cv2.createTrackbar("Lower S", "Trackbars", 50, 255, nothing)
    cv2.createTrackbar("Lower V", "Trackbars", 50, 255, nothing)
    cv2.createTrackbar("Upper H", "Trackbars", 179, 179, nothing)
    cv2.createTrackbar("Upper S", "Trackbars", 255, 255, nothing)
    cv2.createTrackbar("Upper V", "Trackbars", 255, 255, nothing)

    while True:
        # Get the current positions of the trackbars
        lower_h = cv2.getTrackbarPos("Lower H", "Trackbars")
        lower_s = cv2.getTrackbarPos("Lower S", "Trackbars")
        lower_v = cv2.getTrackbarPos("Lower V", "Trackbars")
        upper_h = cv2.getTrackbarPos("Upper H", "Trackbars")
        upper_s = cv2.getTrackbarPos("Upper S", "Trackbars")
        upper_v = cv2.getTrackbarPos("Upper V", "Trackbars")


        # Define HSV bounds
        lower_bound = np.array([lower_h, lower_s, lower_v])
        upper_bound = np.array([upper_h, upper_s, upper_v])
        # Create a mask based on the current HSV bounds
        mask = cv2.inRange(image_hsv, lower_bound, upper_bound)
        kernel = np.ones((5, 5), np.uint8)

        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.erode(mask, kernel)
        mask = cv2.dilate(mask, kernel)


        # Apply the mask to the original image
        result = cv2.bitwise_and(re_image, re_image, mask=mask)

        # Display the mask and the result
        cv2.imshow("Mask", resizeImage(mask,50))
        cv2.imshow("Masked Image", resizeImage(result,50))
        if cv2.waitKey(1) & 0xFF == ord('q'):
        # saved_name = f'{name}_{size}.{ext}'
        # path = os.path.join(out_folder, saved_name)
        # cv2.imwrite(path,left_image)
            break

    # Release the windows
    cv2.destroyAllWindows()
    
'''
    # Define HSV bounds
    lower_bound = np.array([13, 37, 4])
    upper_bound = np.array([179, 255, 255])

    # Create a mask based on the current HSV bounds
    mask = cv2.inRange(image_hsv, lower_bound, upper_bound)
    kernel = np.ones((5, 5), np.uint8)
    
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.erode(mask, kernel)
    mask = cv2.dilate(mask, kernel)


    # Apply the mask to the original image
    result = cv2.bitwise_and(img_map, img_map, mask=mask)
    
    # Display the mask and the result
    # cv2.imshow("plank_image", resizeImage(plank_image,50))
    # cv2.imshow("Mask", resizeImage(mask,50))
    # cv2.imshow("Masked Image", resizeImage(result,50))
    saved_name = f'{name}_{size}_mask{ext}'
    mask_img =  f'mask_{name}_{size}{ext}'
    path_ig = os.path.join(out_folder, saved_name)
    path_m = os.path.join(out_folder, mask_img)
    cv2.imwrite(path_ig,result)
    cv2.imwrite(path_m,mask)

    # Press 'q' to quit
    # if cv2.waitKey(1) & 0xFF == ord('q'):
    #     # saved_name = f'{name}_{size}.{ext}'
    #     # path = os.path.join(out_folder, saved_name)
    #     # cv2.imwrite(path,left_image)
    #     break

    # Release the windows
    # cv2.destroyAllWindows()
