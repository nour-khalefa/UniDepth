import os

import numpy as np
import torch
from PIL import Image

from .dataset import BaseDataset
import pandas as pd
class DrivingStereoDataset(BaseDataset):
    
    min_depth = 0.01
    max_depth = 80
    def __init__(
        self,
        test_mode,
        base_path,
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
        
        
       

        self.cam_intrinsics = self.extract_all_k_101(os.path.join(base_path,'half-image-calib'))
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
        # Use this method to fetch the fy value for a given key
        return self.fy_values.get(key, None)

    def extract_k_101(self,file_path):
        with open(file_path, 'r') as file:
            lines = file.readlines()
            for line in lines:
                if line.startswith("K_101"):
                    values = line.split()[1:]  # Skip the "K_101" part
                    matrix = [float(value) for value in values]
                    tensor = torch.tensor(matrix).reshape(3, 3)
                    return tensor
        return None

    def extract_all_k_101(self,directory):
        cam_intrinsic = {}
        for filename in os.listdir(directory):
            if filename.endswith(".txt"):
                file_path = os.path.join(directory, filename)
                tensor = self.extract_k_101(file_path)
                if tensor is not None:
                    cam_intrinsic[filename.replace(".txt", "")] = tensor
        return cam_intrinsic


    def load_dataset(self):
        
        for _, row in self.data_frame_unique_images.iterrows():
            img_info = dict()
            subset = row['weather']
            image_name = row['image_name']
            mask_label = row['mask_label']
            height = row['height']
            img_info["subset"] = row['weather']
            img_info["image_filename"] = os.path.join(self.base_path, "train-left-image",f"{subset}", image_name)
            img_info["annotation_filename_depth"] = os.path.join(self.base_path, "train-depth-map",f"{subset}", image_name.replace("jpg", "png"))
            
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
            depth = (
                np.asarray(
                    Image.open(
                        self.dataset[idx]["annotation_filename_depth"]
                    )
                ).astype(np.float32)
                / self.depth_scale
            )
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
