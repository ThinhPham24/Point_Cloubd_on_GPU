
# import cv2
# import numpy as np
# import time


# def main():
#     # Check if CUDA is available
#     if not cv2.cuda.getCudaEnabledDeviceCount():
#         print("CUDA is not available. Please ensure you have a CUDA-capable GPU and the correct drivers installed.")
#         return

#     # Initialize CUDA device
#     device_count = cv2.cuda.getCudaEnabledDeviceCount()
#     print(f"Number of CUDA-capable GPUs: {device_count}")
#     # print(f"cv2.getBuildInformation(): {cv2.getBuildInformation()}")
#     # Read the image
#     image_path = '/home/airlab/Desktop/Thinh/000000000034.jpg'
#     image = cv2.imread(image_path,0)
#     if image is None:
#         print("Error: Image not found.")
#         return

#     # Function to perform GPU processing on an image
#     def process_image_on_gpu(image, device_id):
#         cv2.cuda.setDevice(device_id)
#          # Measure performance
#         start_time = time.time()
#         # Upload the image to the GPU
#         gpu_image = cv2.cuda_GpuMat()
#         gpu_image.upload(image)
        


#         # Apply Gaussian blur on the GPU
#         gpu_blur = cv2.cuda.createGaussianFilter(
#             gpu_image.type(), -1, (15, 15), 0)
#         gpu_blurred_image = gpu_blur.apply(gpu_image)
        
#         # Download the result back to the CPU
#         result_image = gpu_blurred_image.download()
#         total_gpu_time = time.time() - start_time

#         # Display processing time
#         print(f"Total GPU  11111 Processing Time: {total_gpu_time:.6f} seconds")
#         return result_image

#     # Measure performance
#     start_time = time.time()

#     # Process image on GPU 0
#     result_image_gpu0 = process_image_on_gpu(image, 0)

#     # If multiple GPUs are available, process on GPU 1 as well
#     if device_count > 1:
#         result_image_gpu1 = process_image_on_gpu(image, 1)

#     total_gpu_time = time.time() - start_time

#     # Display processing time
#     print(f"Total GPU Processing Time: {total_gpu_time:.6f} seconds")

#     # Display the original and blurred images
#     cv2.imshow('Original Image', image)
#     cv2.imshow('Blurred Image (GPU 0)', result_image_gpu0)
#     if device_count > 1:
#         cv2.imshow('Blurred Image (GPU 1)', result_image_gpu1)
#     cv2.waitKey(0)
#     cv2.destroyAllWindows()


# if __name__ == "__main__":
#     main()
# import numpy as np
# import cv2 as cv
# import time

# npTmp = np.random.random((1024, 1024)).astype(np.float32)

# npMat1 = np.stack([npTmp,npTmp],axis=2)
# npMat2 = npMat1
# start_time = time.time()
# cuMat1 = cv.cuda_GpuMat()
# cuMat2 = cv.cuda_GpuMat()
# cuMat1.upload(npMat1)
# cuMat2.upload(npMat2)


# cv.cuda.gemm(cuMat1, cuMat2,1,None,0,None,1)

# print("CUDA using GPU --- %s seconds ---" % (time.time() - start_time))

# start_time = time.time()
# cv.gemm(npMat1,npMat2,1,None,0,None,1)

# print("CPU --- %s seconds ---" % (time.time() - start_time))
# import cv2
# import numpy as np



# import cv2
# import numpy as np
# # from matplotlib import pyplot as plt
def resizeImage(img, percent):
    """
    Resize an image by a given percentage.

    Args:
        img (numpy.ndarray): Input image to be resized.
        percent (float): Percentage by which to resize the image.

    Returns:
        numpy.ndarray: Resized image.
    """
    width = int(img.shape[1] * percent / 100)
    height = int(img.shape[0] * percent / 100)
    dim = (width, height)
    resized = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)
    return resized
# # Load the image
# image = cv2.imread('/home/airlab/Desktop/Thinh/L_remap.png')
# image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# # Convert to HSV color space
# hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

# # Define color range for masking (you may need to adjust these values)
# lower_bound = np.array([120, 30, 50])   # Lower bound for orchid color
# upper_bound = np.array([160, 255, 255]) # Upper bound for orchid color

# # Create mask
# mask = cv2.inRange(hsv_image, lower_bound, upper_bound)

# # Apply morphological operations to clean up the mask
# kernel = np.ones((5, 5), np.uint8)
# mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
# mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

# # Optional: Mask the image to see the result
# masked_image = cv2.bitwise_and(image_rgb, image_rgb, mask=mask)

# # Display the mask and masked image
# plt.figure(figsize=(10, 5))
# plt.subplot(1, 2, 1)
# plt.title("Mask")
# plt.imshow(mask, cmap="gray")
# plt.subplot(1, 2, 2)
# plt.title("Masked Image")
# plt.imshow(masked_image)
# plt.show()
# cv2.imshow("Image", resizeImage(masked_image,50))
# cv2.waitKey(0)
import cv2
import numpy as np
import os
def nothing(x):
    pass

# Load the image
image = cv2.imread('/home/airlab/Desktop/Thinh/pcgeneration/L_13.png')
image_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

# image_hsv = resizeImage(image_hsv,50)
# Create a window
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
    result = cv2.bitwise_and(image, image, mask=mask)
    
    # Display the mask and the result
    # cv2.imshow("plank_image", resizeImage(plank_image,50))
    cv2.imshow("Mask", resizeImage(mask,20))
    cv2.imshow("Masked Image", resizeImage(result,20))

    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        crop_out_folder = '/home/airlab/Desktop/Thinh'
        image_L_file = 'mask_R.png'
        image_R_file = 'mask_color_R.png'
        left_path = os.path.join(crop_out_folder, image_L_file)
        cv2.imwrite(left_path,mask)
        left_path = os.path.join(crop_out_folder, image_R_file)
        cv2.imwrite(left_path,result)
        break

# Release the windows
cv2.destroyAllWindows()