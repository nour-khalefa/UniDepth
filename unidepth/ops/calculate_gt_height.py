import cv2
import numpy as np
import os
import csv

# Function to calculate the height of an object given its binary mask and depth map
import torch

def calculate_height(binary_mask, depth_map, fy):
    # Ensure the binary mask is binary
    binary_mask = binary_mask.float() / 255.0
    depth_map = depth_map.squeeze(0)
    # Extract the depth values of the object using the binary mask
    object_depth_values = depth_map[binary_mask == 1]

    # Filter out zero depth values
    non_zero_depth_values = object_depth_values[object_depth_values > 0]
    if non_zero_depth_values.numel() == 0:
        return None

    # Calculate the median of the non-zero depth values
    D = torch.median(non_zero_depth_values)
    
    # Calculate the height of the object in pixels
    object_pixel_coords = torch.nonzero(binary_mask, as_tuple=False)
    height_in_pixels = torch.max(object_pixel_coords[:, 0]) - torch.min(object_pixel_coords[:, 0])
    
    # Calculate the real height of the object
    real_height = (height_in_pixels.float() * D) / fy

    return real_height.item()

# Paths to the folders
# root_dirctory='/home/nour.khalefa/datasets/DrivingStereo/train'
# subsets = ['2018-07-09-16-11-56','2018-07-10-09-54-03','2018-07-16-15-18-53','2018-07-16-15-37-46','2018-07-18-10-16-21',
# '2018-07-18-11-25-02','2018-07-24-14-31-18','2018-07-27-11-39-31','2018-07-31-11-07-48','2018-07-31-11-22-31',
# '2018-08-13-15-32-19','2018-08-13-17-45-03','2018-08-17-09-45-58','2018-10-10-07-51-49','2018-10-11-17-08-31',
# '2018-10-12-07-57-23','2018-10-15-11-43-36','2018-10-16-07-40-57','2018-10-16-11-13-47','2018-10-16-11-43-02','2018-10-17-14-35-33',
# '2018-10-17-15-38-01','2018-10-18-10-39-04','2018-10-18-15-04-21','2018-10-19-09-30-39','2018-10-19-10-33-08','2018-10-22-10-44-02']
if __name__ == "__main__":
    root_dirctory = ''
    subsets=['']
    output_csv = 'output_heights.csv'

    # Camera intrinsic parameters (example values, replace with actual parameters)
    fy = 2.060674e+03  # Vertical focal length

    # Prepare the CSV file
    with open(output_csv, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['weather','image_name','mask_file_name', 'mask_label', 'height'])
        for subset in subsets:
            original_images_folder = os.path.join(root_dirctory,'left-image-half-size',subset)
            depth_maps_folder = os.path.join(root_dirctory,'depth-map-half-size',subset)
            binary_masks_folder = 'output_masks/'+subset+'_output_masks'
        # Iterate over the images in the original images folder
            for image_name in os.listdir(original_images_folder):
                image_path = os.path.join(original_images_folder, image_name)
                depth_map_path = os.path.join(depth_maps_folder, image_name.replace('.jpg', '.png'))

                # Load the original image and depth map
                original_image = cv2.imread(image_path)
                depth_map = cv2.imread(depth_map_path, cv2.IMREAD_UNCHANGED)

                # Convert depth map from uint16 to float and divide by 256
                depth_map = depth_map.astype(np.float32) / 256.0

                # Find corresponding binary masks
                mask_files = [f for f in os.listdir(binary_masks_folder) if f.startswith(image_name.replace('.jpg', ''))]

                if not mask_files:
                    writer.writerow([subset,image_name, '', '',''])
                else:
                    for mask_file in mask_files:
                        mask_path = os.path.join(binary_masks_folder, mask_file)
                        binary_mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

                        # Calculate the height of the object
                        height = calculate_height(binary_mask, depth_map, fy)

                        # Extract the mask label from the mask file name
                        mask_label = mask_file.split('_')[-2]

                        writer.writerow([subset,image_name,mask_file, mask_label, height if height is not None else ''])

    print(f"Height calculations saved to {output_csv}")