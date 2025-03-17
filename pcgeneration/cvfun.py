import open3d as o3d
import cv2
import numpy as np
from PIL import Image
def show(pcPoints, pcColors):
    """
    Visualizes the generated 3D point cloud using Open3D.
    """
    if pcPoints is not None and pcColors is not None:
        points = pcPoints.reshape(-1, 3)
        colors = pcColors.reshape(-1, 3) 
        colors = np.asarray(colors/255) # rescale to 0 to 1
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        pcd.colors = o3d.utility.Vector3dVector(colors)
        o3d.visualization.draw_geometries([pcd])
    else:
        print("No 3D points or colors to visualize.")
def write(pathName,points, colors):
    ply_header = '''ply
format ascii 1.0
element vertex %(vert_num)d
property float x
property float y
property float z
property uchar red
property uchar green
property uchar blue
end_header
'''
    verts = points.reshape(-1, 3)
    colors = colors.reshape(-1, 3) 
    verts = np.hstack([verts, colors])
    with open(pathName, 'wb') as f:
        f.write((ply_header % dict(vert_num=len(verts))).encode('utf-8'))
        np.savetxt(f, verts, fmt='%f %f %f %d %d %d ')
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
def read_stereo_paremater(path: str):
    get_parameter = cv2.FileStorage(path, cv2.FileStorage_READ)
    Q = get_parameter.getNode('q').mat()
    R_x = get_parameter.getNode('stereoMapR_x').mat()
    R_y = get_parameter.getNode('stereoMapR_y').mat()
    L_x = get_parameter.getNode('stereoMapL_x').mat()
    L_y = get_parameter.getNode('stereoMapL_y').mat()

    get_parameter.release()
    return np.array([R_x, R_y, L_x, L_y, Q], dtype=object)
def read_img(path_img: str, color = True):
    color_img  = cv2.imread(path_img)
    gray_color = cv2.cvtColor(color_img, cv2.COLOR_BGR2GRAY)
    if color:
        re_img = color_img
    else:
        re_img = gray_color
    return re_img
def remap_img(image, map1, map2):
    rectified_image = cv2.remap(image, map1, map2, interpolation=cv2.INTER_LINEAR)
    return rectified_image
def remove_leaf(left_image_path):
    image_gray = read_img(left_image_path, False)
    # image_color = read_img(left_image_path, True)
    stpoint = [1081, 948] # x, y - w, h
    h, w = image_gray.shape
    plank_img = np.ones_like(image_gray)
    plank_img[stpoint[1]:, :] = image_gray[stpoint[1]:, :]
    kernel = np.ones((11, 11), np.uint8) 
    img_erosion = cv2.erode(plank_img, kernel, iterations=3) 
    img_dilation = cv2.dilate(img_erosion, kernel, iterations=3) 
    sub_img = cv2.subtract(image_gray, img_dilation)
    final_img = np.ones_like(image_gray)
    contours, _ = cv2.findContours(sub_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    max_cnt = max(contours, key=cv2.contourArea)
    cv2.drawContours(final_img, [max_cnt], -1, (255), thickness=cv2.FILLED)
    sub_img2 = cv2.bitwise_and(image_gray, final_img)
    return sub_img2
def show_2d(win_name, image, wait_time, scale = 50):
    if isinstance(image, np.ndarray):
        image = image
    elif isinstance(image, str):
        # Input is a file path, read the image first
        image = cv2.imread(image)
    resize_img = resizeImage(image, scale)
    cv2.imshow(f"{win_name}", resize_img)
    cv2.waitKey(wait_time)
    cv2.destroyAllWindows()
def specklefilter_dis(dis_map, newVal=0, maxSpeckleSize=200, maxDiff=16):
    """
    Apply speckle filtering to a disparity map to remove noise.

    Args:
        dis_map (numpy.ndarray): The input disparity map to be filtered.
        newVal (int, optional): The value to replace detected speckles. Default is 0.
        maxSpeckleSize (int, optional): Maximum size of the speckles to be removed. Default is 25.
        maxDiff (int, optional): Maximum disparity difference to consider a pixel part of the same speckle. Default is 16.

    Returns:
        numpy.ndarray: The filtered disparity map.
    """
    # Check if the disparity map is of type int16, convert if not
    if dis_map.dtype != np.int16:
        dis_map = dis_map.astype(np.int16)
    
    # Apply speckle filtering using OpenCV's filterSpeckles function
    cv2.filterSpeckles(dis_map, newVal=newVal, maxSpeckleSize=maxSpeckleSize, maxDiff=maxDiff)
    
    # Convert the disparity map back to float32 and normalize it by dividing by 16
    filtered_dis = dis_map.astype(np.float32) / 16
    
    # Return the filtered disparity map
    return filtered_dis
def split_and_pad_images(left_image, right_image, num_parts=3, padding=256):
    """
    Split the left and right images into a specified number of parts 
    (with padding applied) and return the parts as lists.

    Args:
        left_image (numpy.ndarray): The left image to be split and padded.
        right_image (numpy.ndarray): The right image to be split and padded.
        num_parts (int, optional): The number of parts to split the images into (default is 3).
        padding (int, optional): The padding to apply to the image parts. Default is 256.

    Returns:
        tuple: A tuple containing two lists:
            - agg_l: List of padded parts of the left image.
            - agg_r: List of padded parts of the right image.
    """
    # Get the dimensions of the original image
    height, width = left_image.shape

    # Calculate the height of each part
    crop_height = height // num_parts  # Split height into `num_parts` parts

    # Initialize lists to store the cropped and padded parts
    agg_l = []
    agg_r = []

    for i in range(num_parts):
        # Calculate start and end indices for each crop
        start_row = i * crop_height
        end_row = (i + 1) * crop_height if (i + 1) < num_parts else height

        # Cropped parts, with padding applied
        if i == 0:
            left_part = left_image[start_row:end_row + padding, :]
            right_part = right_image[start_row:end_row + padding, :]
        else:
            left_part = left_image[start_row - padding:end_row + padding, :]
            right_part = right_image[start_row - padding:end_row + padding, :]
        # Append padded parts to the lists
        agg_l.append(left_part)
        agg_r.append(right_part)
    return agg_l, agg_r
def assemble_disparity_map(merge_dis, height, width, num_parts, padding):
    """
    Assemble the disparity map from different cropped parts.

    Args:
        merge_dis (list of numpy.ndarray): List of cropped disparity map parts.
        height (int): The height of the final disparity map.
        width (int): The width of the final disparity map.
        num_parts (int): Number of parts the image is divided into (3, 4, or 5).
        padding (int): Padding value to be used when placing the parts back.

    Returns:
        numpy.ndarray: Final assembled disparity map.
    """
    # Calculate the height of each part
    crop_height = height // num_parts
    mindis = np.min(merge_dis[1])
    # Initialize the final disparity map with zeros
    final_dis = np.zeros((height, width), dtype=np.uint8)*mindis
    for i in range(num_parts):
        # Calculate start and end indices for each crop
        start_row = i * crop_height
        end_row = (i + 1) * crop_height if (i + 1) < num_parts else height
        # Cropped parts, with padding applied
        if i == 0:
            final_dis[start_row:end_row + padding, :] = merge_dis[i]
        else:
            final_dis[start_row - padding:end_row + padding, :] = merge_dis[i]
    return final_dis
def apply_hsv_mask(image, lower_bound=[13, 37, 4], upper_bound=[179, 255, 255], kernel_size=(5, 5)):
    """
    Apply an HSV mask to the input image to isolate colors within a specified range.

    Args:
        image (numpy.ndarray): The input image in BGR format.
        lower_bound (list): Lower HSV bounds for masking.
        upper_bound (list): Upper HSV bounds for masking.
        kernel_size (tuple): Size of the morphological operation kernel.

    Returns:
        numpy.ndarray: The masked image with colors outside the HSV range removed.
    """
    # Convert image to HSV color space
    image_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Define HSV bounds
    lower_bound = np.array(lower_bound)
    upper_bound = np.array(upper_bound)

    # Create a mask based on the HSV bounds
    mask = cv2.inRange(image_hsv, lower_bound, upper_bound)
    kernel = np.ones(kernel_size, np.uint8)

    # Apply morphological operations to clean up the mask
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.erode(mask, kernel)
    mask = cv2.dilate(mask, kernel)

    # Apply the mask to the original image
    result = cv2.bitwise_and(image, image, mask=mask)
    return result
def compute_point_cloud_from_stereo(stereo, left_image, right_image, left_image_color, mask_L=None, Q=None, re_background=192):
    """
    Computes a 3D point cloud from stereo images using CUDA-accelerated disparity computation.

    Args:
        stereo (cv2.cuda.StereoSGM): CUDA-based stereo matching object for computing disparity.
        left_image (numpy.ndarray): Grayscale left image used for disparity computation.
        right_image (numpy.ndarray): Grayscale right image used for disparity computation.
        left_image_color (numpy.ndarray): Color left image for assigning colors to the 3D point cloud.
        mask_L (numpy.ndarray, optional): Mask to apply on the computed disparity map.
        Q (numpy.ndarray, optional): Reprojection matrix for transforming disparity to 3D points.
        re_background (int): Background threshold value to filter out low-depth areas for cleaner point clouds.

    Returns:
        tuple: Contains 3D points (numpy.ndarray) and their corresponding RGB colors (numpy.ndarray) 
               for the generated point cloud.
    """
    # Upload grayscale images to GPU for processing
    left_gpu = cv2.cuda_GpuMat()
    right_gpu = cv2.cuda_GpuMat()
    left_gpu.upload(left_image)
    right_gpu.upload(right_image)

    # Compute the disparity map using the CUDA Stereo SGM algorithm
    disparity_gpu = stereo.compute(left_gpu, right_gpu)
    disparity_map = disparity_gpu.download()

    # Apply speckle filtering to remove noise and normalize the disparity map
    disparity_filtered = specklefilter_dis(disparity_map)
    depth = cv2.bitwise_and(disparity_filtered, disparity_filtered, mask=mask_L)

    # Release GPU resources
    left_gpu.release()
    right_gpu.release()
    stereo.release()

    # Reproject disparity map to 3D coordinates using the Q matrix, or create one if Q is not provided
    h, w = left_image.shape[:2]
    Q1 = np.float32([[1, 0, 0, -w / 2.0],
                    [0, -1, 0, h / 2.0],
                    [0, 0, 0, -Q[2, 3]],
                    [0, 0, Q[3, 2], Q[3, 3]]])
    pcPoints = cv2.reprojectImageTo3D(depth, Q1)
    # Extract color information and apply background threshold to mask out low-depth regions
    pcColors = cv2.cvtColor(left_image_color, cv2.COLOR_BGR2RGB)
    mask_ = depth > depth.min() + re_background
    pcPoints = pcPoints[mask_]
    pcColors = pcColors[mask_]
    return pcPoints, pcColors

def compute_point_cloud_with_padding(stereo, left_image, right_image, left_image_color, mask_L=None, Q=None, padding= 256, split_parts=3, re_background = 192):
    """
    Computes a 3D point cloud from stereo images, with support for image padding and splitting 
    for improved disparity calculation.

    Args:
        stereo (cv2.cuda.StereoSGM): Stereo matching object for computing disparity.
        left_image (numpy.ndarray): Grayscale left image for disparity calculation.
        right_image (numpy.ndarray): Grayscale right image for disparity calculation.
        left_image_color (numpy.ndarray): Color left image, used for coloring the 3D point cloud.
        mask_L (numpy.ndarray, optional): Binary mask to apply to the computed disparity map.
        Q (numpy.ndarray): Reprojection matrix for transforming disparity to 3D points.
        padding (int): Padding size for each split image segment to improve disparity at edges.
        split_parts (int): Number of parts to split each image for separate processing.

    Returns:
        tuple: Contains 3D points (numpy.ndarray) and their corresponding RGB colors (numpy.ndarray) 
               for the generated point cloud.
    """
    # Retrieve dimensions of the color image
    h, w = left_image_color.shape[:2]
    height_, width_ = left_image.shape

    # Ensure images are loaded correctly
    if left_image is None or right_image is None:
        raise ValueError("Could not open or find the images!")

    # Split and pad the left and right images into multiple segments for individual processing
    agg_l, agg_r = split_and_pad_images(left_image, right_image, split_parts, padding)
    merge_dis = []

    # Process each segment to compute disparity maps
    for im_l, im_r in zip(agg_l, agg_r):
        # Upload image segments to GPU memory
        left_gpu = cv2.cuda_GpuMat()
        right_gpu = cv2.cuda_GpuMat()
        left_gpu.upload(im_l)
        right_gpu.upload(im_r)

        # Compute disparity map for each segment using StereoSGM on the GPU
        disparity_gpu = stereo.compute(left_gpu, right_gpu)
        disparity_map = disparity_gpu.download()
        merge_dis.append(disparity_map)

    # Merge segmented disparity maps into a single final disparity map
    final_dis = assemble_disparity_map(merge_dis, height_, width_, split_parts, padding)
    filtered_disp_vis = specklefilter_dis(final_dis)

    # Apply mask and reproject disparity map to 3D space using Q matrix
    depth = cv2.bitwise_and(filtered_disp_vis, filtered_disp_vis, mask=mask_L)
    Q1 = np.float32([[1, 0, 0, -w / 2.0],
                     [0, -1, 0, h / 2.0],
                     [0, 0, 0, -Q[2, 3]],
                     [0, 0, Q[3, 2], Q[3, 3]]])
    pcPoints = cv2.reprojectImageTo3D(depth, Q1)

    # Extract color information and apply mask for valid depth points
    pcColors = cv2.cvtColor(left_image_color, cv2.COLOR_BGR2RGB)
    mask_ = depth > depth.min() + re_background
    pcPoints = pcPoints[mask_]
    pcColors = pcColors[mask_]

    return pcPoints, pcColors
