from unidepth.models import UniDepthV1HF
import numpy as np
from PIL import Image
import torch
from unidepth.utils import colorize, image_grid
import unidepth.dataloaders as custom_dataset
from torch.utils.data import DataLoader, Dataset
import unidepth.ops as ops
from unidepth.utils.geometric import generate_rays
from einops import rearrange
import cv2
import config
def custom_collate_fn(batch):
    # Stack the images and depths
    images = torch.stack([item['image'] for item in batch])
    depths = torch.stack([item['depth'] for item in batch if item['depth'] is not None])
    
    # Collect masks as a list of lists of tensors
    masks = [item['masks'] for item in batch]
    
    # Collect heights and intrinsics
    heights = [item['heights'] for item in batch]
    intrinsics = torch.stack([item['K'] for item in batch])

    return {
        'image': images,
        'depth': depths,
        'masks': masks,
        'heights': heights,
        'K': intrinsics
    }
if __name__ == "__main__":
    # # Define the path to the CSV file and the root directory
    # csv_file1 = 'output_heights2.csv'
    # root_dir1 = 'output_masks'

    # csv_file2 = '/scratch/32035138/cityscapes_train_output_heights.csv'
    # root_dir2 = '/scratch/32035138/output_masks'

    # save_dir = 'augmented_images'
    # plot_save_path = 'loss_plot.png'
    # best_model_path = 'best_model.pth'
    # latest_model_path = 'latest_model.pth'

    # # Load the label_to_index mapping from the file
    # with open('label_to_index.json', 'r') as f:
    #     label_to_index = json.load(f)
    # Initialize the dataset and dataloader
    

    # Create an instance of the KITTIDataset
    # dataset =  getattr(custom_dataset,'KITTIDataset')(
    #     test_mode=False,  # Set to True if you are in test mode
    #     base_path='/home/nour.khalefa/datasets/kitti',  # Replace with the actual path to your dataset
    #     depth_scale=256,
    #     crop='eigen',  # or 'garg' depending on your requirement
    #     is_dense=False,
    #     benchmark=False,
    #     augmentations_db={},  # Add any augmentations if needed
    #     normalize=True
    # )

    # Create an instance of the DrivingStereoDataset
    #dataset =  getattr(custom_dataset,'DrivingStereoDataset')(
    batch_size = 2
    cityscapes_dataset_train = getattr(custom_dataset,'CityScapesDataset')(
        test_mode=False,  # Set to True if you are in test mode
        #base_path='/home/nour.khalefa/datasets/DrivingStereo/train',  # Replace with the actual path to your dataset
        base_path=config.CITYSCAPES_BASE_PATH,
        csv_path=config.CITYSCAPES_CSV_TRAIN,
        depth_scale=256,
        # crop='eigen',  # or 'garg' depending on your requirement
        is_dense=False,
        benchmark=False,
        augmentations_db={},  # Add any augmentations if needed
        normalize=True
    )

    cityscapes_dataset_val = getattr(custom_dataset,'CityScapesDataset')(
        test_mode=True,  # Set to True if you are in test mode
        base_path=config.CITYSCAPES_BASE_PATH,
        csv_path=config.CITYSCAPES_CSV_VAL,
        split='val',
        depth_scale=256,
        # crop='eigen',  # or 'garg' depending on your requirement
        is_dense=False,
        benchmark=False,
        augmentations_db={},  # Add any augmentations if needed
        normalize=True
    )


    drvingstereo_dataset_train = getattr(custom_dataset,'DrivingStereoDataset')(
        test_mode=False,  # Set to True if you are in test mode
        base_path=config.DRIVINGSTEREO_BASE_PATH_TRAIN, 
        csv_path=DRIVINGSTEREO_CSV_TRAIN,
        depth_scale=256,
        # crop='eigen',  # or 'garg' depending on your requirement
        is_dense=False,
        benchmark=False,
        augmentations_db={},  # Add any augmentations if needed
        normalize=True
    )

    drvingstereo_dataset_val = getattr(custom_dataset,'DrivingStereoDataset')(
        test_mode=True,  # Set to True if you are in test mode
        base_path=DRIVINGSTEREO_BASE_PATH_VAL,
        csv_path=DRIVINGSTEREO_CSV_VAL,
        split='val',
        depth_scale=256,
        # crop='eigen',  # or 'garg' depending on your requirement
        is_dense=False,
        benchmark=False,
        augmentations_db={},  # Add any augmentations if needed
        normalize=True
    )
    
    # Create a DataLoader
  # Optional: Create merged DataLoaders for training and validation
    merged_train_dataset = ConcatDataset([cityscapes_dataset_train, drivingstereo_dataset_train])
    merged_val_dataset = ConcatDataset([cityscapes_dataset_val, drivingstereo_dataset_val])

    merged_dataloader_train = DataLoader(
        merged_train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=1,
        collate_fn=custom_collate_fn
    )

    merged_dataloader_val = DataLoader(
        merged_val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=1,
        collate_fn=custom_collate_fn
    )
    model = UniDepthV1HF.from_pretrained("nielsr/unidepth-v1-convnext-large")

    # Freeze all parameters except those in DepthHead
    for name, param in model.named_parameters():
        if "depth_layer" not in name:
            param.requires_grad = False

    # Move to CUDA, if any
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-6)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
    depth_criterion = ops.SILog(weight=0.15)
    height_loss_fn = ops.HeightLoss().to('cuda')
    # Example of iterating through the dataloader
    for batch in dataloader:
        optimizer.zero_grad()
        data = {'image': batch['image'].to(device), 'K': batch['K'].to(device)}
        print("done loading")
        # angles,intrinsics,points,depth = model(data, {})
        outputs = model(data, {})
        depth = outputs['depth']
        
        rays, angles = generate_rays(
                batch['K'], [462,616], noisy=False
            )
        # angles = rearrange(angles, "b (h w) c -> b c h w", h=462, w=616)
        # print(angles,np.shape(angles))
        # print(outputs['angles'][0],np.shape(outputs['angles'][0]))
        
        loss1 = depth_criterion(depth, batch['depth'].to(device))
        


        
        height_target = [torch.tensor(height).to('cuda') for height in batch['heights']]  # Ensure height_target is on GPU
        masks = [[mask.to(device) for mask in masks] for masks in batch['masks']]
        # Calculate the loss
        loss2 = height_loss_fn(outputs['depth'], height_target, masks,fy=batch['fy'])
        loss = loss1+loss2
        #print(loss)
        loss.backward()
        optimizer.step()


        # #test mask and predicted depth
        # mask = batch['masks'][0].squeeze().cpu().detach().numpy()
        # depth = outputs['depth'].squeeze().cpu().detach().numpy()
        # print(np.shape(depth))
        # print(mask)
        # # mask = mask.astype(np.uint8)

        # # Apply the mask to the depth map
        # masked_depth = depth * mask
        # print(np.min(depth),np.max(depth))
        # # Normalize the depth map for visualization
        # # normalized_depth = cv2.normalize(masked_depth, None, 0, 255, cv2.NORM_MINMAX)
        # # depth = cv2.normalize(depth, None, 0, 255, cv2.NORM_MINMAX)
        # for mask in batch['masks']:
        #     mask = mask.squeeze().cpu().detach().numpy()
        #     height = ops.calculate_height(mask, depth, 2.060674e+03)
        #     print(height)
        
        # Example usage
    
    

        # Assuming depth_pred, height_target, and masks are already defined and moved to the GPU
        

        
        






            # image_path = "assets/driving_stereo/drivingstereo_test_img.png"
    # depth_gt= None
    # # Load the RGB image and the normalization will be taken care of by the model
    # rgb = torch.from_numpy(np.array(Image.open(image_path))).permute(2, 0, 1) # C, H, W
    # transform = transforms.Compose([
    #     transforms.RandomHorizontalFlip(),
    #     transforms.Resize((224, 224)),
    #     transforms.ToTensor(),
    #     transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # MobileNetV2 preprocessing
    # ])

    # dataset1 = CustomDataset(csv_file1, root_dir1, transform=transform,label_to_index=label_to_index)
    # dataset2 = CustomDataset(csv_file2, root_dir2, transform=transform,label_to_index=label_to_index)

    # # Set the seed for reproducibility of the dataset split
    # seed = 42
    # generator = torch.Generator().manual_seed(seed)
    # # Split the dataset into training and validation sets (70/30 split)
    # train_size1 = int(0.7 * len(dataset1))
    # val_size1 = len(dataset1) - train_size1
    # train_dataset1, val_dataset1 = random_split(dataset1, [train_size1, val_size1], generator=generator)

    # train_size2 = int(0.7 * len(dataset2))
    # val_size2 = len(dataset2) - train_size2
    # train_dataset2, val_dataset2 = random_split(dataset2, [train_size2, val_size2], generator=generator)

    # # Combine the training and validation sets
    # train_dataset = ConcatDataset([train_dataset1, train_dataset2])
    # val_dataset = ConcatDataset([val_dataset1, val_dataset2])
    # # Set num_workers to 4 for parallel data loading
    # num_workers = 4

    # train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=num_workers)
    # val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False, num_workers=num_workers)

    # # Save augmented images
    # #save_augmented_images(dataset, save_dir, num_images=10)

    # # Initialize the model
    # num_labels = dataset1.num_labels
    # model = MobileNetV2ForRegression(num_labels=num_labels)
    # print("break label was saved")
    # # Move model to GPU if available
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # model.to(device)

    # # Define optimizer, loss function, and learning rate scheduler
    # optimizer = torch.optim.Adam(model.parameters(), lr=1e-6)
    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
    # criterion = nn.MSELoss()

    # # Load the latest model if it exists
    # if os.path.exists(latest_model_path):
    #     model.load_state_dict(torch.load(latest_model_path))
    #     print("Loaded latest model from checkpoint.")

    # # Training loop
    # num_epochs = 50
    # train_losses = []
    # val_losses = []
    # best_val_loss = float('inf')
    # print("begin training")
    # for epoch in range(26,num_epochs):
    #     model.train()
    #     train_loss = 0.0
    #     for images, labels, heights in train_loader:
    #         images, labels, heights = images.to(device), labels.to(device), heights.to(device)
    #         optimizer.zero_grad()
    #         outputs = model(images, labels)
    #         loss = criterion(outputs.squeeze(), heights)
    #         loss.backward()
    #         optimizer.step()
    #         train_loss += loss.item()
    #     train_loss /= len(train_loader)
    #     train_losses.append(train_loss)
    #     print("after training loop")
    #     # Validation loop
    #     model.eval()
    #     val_loss = 0.0
    #     with torch.no_grad():
    #         for images, labels, heights in val_loader:
    #             images, labels, heights = images.to(device), labels.to(device), heights.to(device)
    #             outputs = model(images, labels)
    #             loss = criterion(outputs.squeeze(), heights)
    #             val_loss += loss.item()
        
    #     val_loss /= len(val_loader)
    #     val_losses.append(val_loss)

    #     print(f"Epoch {epoch+1}, Train Loss: {train_loss}, Validation Loss: {val_loss}")

    #     # Save the latest model
    #     torch.save(model.state_dict(), latest_model_path)
    #     print(f"Saved latest model at epoch {epoch+1}")

    #     # Save the best model if validation loss has decreased
    #     if val_loss < best_val_loss:
    #         best_val_loss = val_loss
    #         torch.save(model.state_dict(), best_model_path)
    #         print(f"Saved best model at epoch {epoch+1} with validation loss: {val_loss}")

    # # Plot the training and validation losses
    # plt.figure()
    # plt.plot(range(1, num_epochs + 1), train_losses, label='Train Loss')
    # plt.plot(range(1, num_epochs + 1), val_losses, label='Validation Loss')
    # plt.xlabel('Epoch')
    # plt.ylabel('Loss')
    # plt.legend()
    # plt.title('Training and Validation Losses')
    # plt.savefig(plot_save_path)
    # plt.show()

    # print("Training and validation completed.")