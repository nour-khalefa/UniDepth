from unidepth.models import UniDepthV1HF
import numpy as np
from PIL import Image
import torch
from unidepth.utils import colorize, image_grid
import config
model = UniDepthV1HF.from_pretrained("nielsr/unidepth-v1-convnext-large")

# Move to CUDA, if any
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
image_path = config.IMG_PATH
depth_gt= None
# Load the RGB image and the normalization will be taken care of by the model
rgb = torch.from_numpy(np.array(Image.open(image_path))).permute(2, 0, 1) # C, H, W

predictions = model.infer(rgb)
# Metric Depth Estimation
depth = predictions["depth"]

# Point Cloud in Camera Coordinate
xyz = predictions["points"]

# Intrinsics Prediction
intrinsics = predictions["intrinsics"]

depth_pred = predictions["depth"].squeeze().cpu().numpy()
# colorize
depth_pred_col = colorize(depth_pred, vmin=0.01, vmax=80.0, cmap="magma_r")
Image.fromarray(depth_pred_col).save("assets/driving_stereo/drivingstereo_test_img_depth.png")
print(intrinsics)

depth_gt = np.array(Image.open(config.DEPTH_GT_PATH)).astype(float) / 256.0

if (True):
    print(depth_gt)
    depth_arel = np.abs(depth_gt - depth_pred) / depth_gt
    depth_arel[depth_gt == 0.0] = 0.0
    print(f"ARel: {depth_arel[depth_gt > 0].mean() * 100:.2f}%")
    depth_gt_col = colorize(depth_gt, vmin=0.01, vmax=80.0, cmap="magma_r")
    Image.fromarray(depth_gt_col).save("assets/driving_stereo/drivingstereo_test_img_depth_gt.png")