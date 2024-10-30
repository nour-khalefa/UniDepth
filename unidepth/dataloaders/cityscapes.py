import os

import numpy as np
import torch
from PIL import Image

from .dataset import BaseDataset
import pandas as pd
import json
class CityScapesDataset(BaseDataset):
    
    min_depth = 0.01
    max_depth = 80
    def __init__(
        self,
        test_mode,
        base_path,  #"/scratch/33014037"
        split='train', # val,test
        csv_path,
        depth_scale=256,
        crop=None,
        is_dense=False,
        benchmark=False,
        augmentations_db={},
        normalize=True,
        **kwargs,
    ):
        super().__init__(test_mode, base_path, benchmark, normalize)
        self.test_mode = test_mode
        self.depth_scale = depth_scale
        self.crop = crop
        self.is_dense = is_dense
        self.height = 462
        self.width = 616
        self.data_split = split
        
       

        self.cam_intrinsics,self.base,self.fx = self.extract_cam_info(os.path.join(base_path,"camera",split))
        self.data_frame = pd.read_csv(csv_path)
        
        # Filter out rows with empty labels or label equal to "traffic light"
        self.data_frame = self.data_frame[self.data_frame['mask_label'].notna()]
        self.data_frame = self.data_frame[self.data_frame['mask_label'] != 'traffic light']
        self.data_frame = self.data_frame[self.data_frame['height'] > 0.4]
        self.data_frame = self.data_frame[self.data_frame['height'] < 5]

        # Replace NaN values in the 'weather' column with an empty string
        self.data_frame['weather'] = self.data_frame['weather'].fillna('')
        self.data_frame_unique_images = self.data_frame.drop_duplicates(subset='image_name', keep='first')
       
         # load annotations
        self.load_dataset()
        for k, v in augmentations_db.items():
            setattr(self, k, v)

        self.fy_values = {key: intrinsics[1, 1].item() for key, intrinsics in self.cam_intrinsics.items()}

    def get_fy(self, key):
        return self.fy_values.get(key, None)

    def extract_cam_info(self,directory_path):
        intrinsics_data = {}
        base_data = {}
        fx_dict={}
        for sub_dir in os.listdir(directory_path):
            sub_dir_path = os.path.join(directory_path, sub_dir)

            if os.path.isdir(sub_dir_path):  # Check if it’s a directory
                json_files = [f for f in os.listdir(sub_dir_path) if f.endswith('.json')]

                if json_files:  # Ensure there's at least one JSON file
                    first_json_file = os.path.join(sub_dir_path, json_files[0])
                    
                    with open(first_json_file, 'r') as file:
                        data = json.load(file)
                        if 'intrinsic' in data:
                            # Extract the intrinsic values
                            fx = data['intrinsic'].get('fx', 0.0)
                            fy = data['intrinsic'].get('fy', 0.0)
                            u0 = data['intrinsic'].get('u0', 0.0)
                            v0 = data['intrinsic'].get('v0', 0.0)
                            
                            # Create a 3x3 matrix
                            matrix = [
                                [fx, 0, u0],
                                [0, fy, v0],
                                [0, 0, 1]
                            ]
                            
                            # Convert the matrix to a torch tensor
                            tensor = torch.tensor(matrix, dtype=torch.float32)
                            intrinsics_data[sub_dir] = tensor
                            base_data[sub_dir] = data['extrinsic']['baseline']
                            fx_dict[sub_dir]=fx

        return intrinsics_data,base_data,fx_dict

    def load_dataset(self):
        
        for _, row in self.data_frame_unique_images.iterrows():
            img_info = dict()
            subset = row['weather']
            image_name = row['image_name']
            mask_label = row['mask_label']
            height = row['height']
            img_info["subset"] = row['weather']
            img_info["image_filename"] = os.path.join(self.base_path, "leftImg8bit",self.data_split,f"{subset}", image_name)
            img_info["annotation_filename_disparity"] = os.path.join(self.base_path, "disparity",self.data_split,f"{subset}", image_name.replace("leftImg8bit", "disparity"))
            
            img_info["camera_intrinsics"] = self.cam_intrinsics[
                    subset
                ]
            
            # Loop through the filtered rows
            img_info["masks"]=[]
            img_info["heights"]=[]
        
            for _, row2 in self.data_frame.loc[self.data_frame['image_name'] == image_name].iterrows(): 
                img_info["masks"].append(os.path.join(self.base_path,'output_masks',f'{subset}_output_masks',row2["mask_file_name"]))
                img_info["heights"].append(row["height"])
            self.dataset.append(img_info)

        print(
            f"Loaded {len(self.dataset)} images."
        )

    def __getitem__(self, idx):
        
        # Load image
        image = np.asarray(
            Image.open(
                self.dataset[idx]["image_filename"]
            )
        ).astype(np.uint8)

        # Load depth map if not in benchmark mode
        depth = None
        if not self.benchmark:
            depth = self.get_depth_from_disparity(self.dataset[idx]["annotation_filename_disparity"],self.dataset[idx]["subset"])
           
        masks = []
        for i in self.dataset[idx]["masks"]:
            masks.append(
                np.asarray(
                    Image.open(
                        i
                    )
                ).astype(np.uint8)
            )

        # Copy info and intrinsics
        info = self.dataset[idx].copy()
        intrinsics = self.dataset[idx]["camera_intrinsics"].clone()
        heights = self.dataset[idx]["heights"]

        # Resize and pad image, and adjust intrinsics
        image, intrinsics, depth,masks = self.resize_and_pad(image, intrinsics,depth,masks)

        # Normalize image
        image = self.normalize(image)

        # Convert to BxCxHxW format
        image = image.transpose((2, 0, 1))
        depth = depth.transpose((2, 0, 1))
        # for idx in range(len(masks)):
        #     masks[idx] = masks[idx].transpose((2, 0, 1))
        # Add batch and channel dimensions
        
        # Prepare the data dictionary
        data = {"image": torch.tensor(image, dtype=torch.float32), "K": intrinsics,"depth":torch.tensor(depth, dtype=torch.float32),
        "masks":[torch.tensor(mask, dtype=torch.float32) for mask in masks],"heights":heights,"fy":self.get_fy(self.dataset[idx]["subset"])}

        # Return the data dictionary
        return data

    def resize_and_pad(self, image, intrinsics,depth,masks):
        target_height, target_width = self.height, self.width

        # Resize image
        height, width, _ = image.shape
        scale = min(target_height / height, target_width / width)
        new_height = int(height * scale)
        new_width = int(width * scale)
        resized_image = np.array(Image.fromarray(image).resize((new_width, new_height), Image.BILINEAR))
        resized_depth = np.array(Image.fromarray(depth).resize((new_width, new_height), Image.NEAREST))
        
        resized_depth = np.expand_dims(resized_depth, axis=2) 
        for idx in range(len(masks)):
            masks[idx] =np.array(Image.fromarray(masks[idx]).resize((new_width, new_height), Image.NEAREST))
            # masks[idx]= np.expand_dims(masks[idx], axis=2)
        # Pad image
        pad_height = target_height - new_height
        pad_width = target_width - new_width
        padded_image = np.pad(resized_image, ((0, pad_height), (0, pad_width), (0, 0)), mode='constant', constant_values=0)
        padded_depth = np.pad(resized_depth, ((0, pad_height), (0, pad_width), (0, 0)), mode='constant', constant_values=0)
        
        for idx in range(len(masks)):
            masks[idx] = np.pad(masks[idx], ((0, pad_height), (0, pad_width)), mode='constant', constant_values=0)
        # Adjust intrinsics
        intrinsics[0, 0] *= scale
        intrinsics[1, 1] *= scale
        intrinsics[0, 2] = intrinsics[0, 2] * scale + pad_width / 2
        intrinsics[1, 2] = intrinsics[1, 2] * scale + pad_height / 2

        return padded_image, intrinsics, padded_depth,masks

    def normalize(self, image):
        imagenet_mean = [0.485, 0.456, 0.406]
        imagenet_std = [0.229, 0.224, 0.225]
        image = image / 255.0
        image = (image - imagenet_mean) / imagenet_std
        return image

#get disparity path and returns depth as np array
    def get_depth_from_disparity(self,disparity_path,subset):
        disparity_map = (
                    np.asarray(
                            Image.open(disparity_path)
                        ).astype(np.float32)
                    )
        # Initialize the depth map with the same shape as the disparity map
        depth_map = np.zeros_like(disparity_map, dtype=np.float32)

        # Compute the depth values
        valid_pixels = disparity_map > 0
        disparity_map[valid_pixels] = (disparity_map[valid_pixels].astype(np.float32) - 1.0) / self.depth_scale
        valid_pixels = disparity_map > 0
        depth_map[valid_pixels] = (self.fx[subset] * self.base[subset]) / disparity_map[disparity_map>0]
        
        return depth_map