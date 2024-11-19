import numpy as np
import matplotlib.pyplot as plt
import os
import random
from scipy import ndimage
import cv2
from datasets import Dataset
from PIL import Image
from torch.optim import Adam
import os
import glob


# Path to mask and image folders
image_folder = 'images'
mask_folder = 'masks'

mask_list = []
image_list = []

# Get list of mask files
mask_files = glob.glob(os.path.join(mask_folder, '*.png'))  # Assuming masks are PNG files

# Iterate through mask files
for mask_file in mask_files:
    # Extract file name without extension
    mask_name = os.path.splitext(os.path.basename(mask_file))[0]

    # Search for corresponding image file
    image_file = os.path.join(image_folder, mask_name + '.png')  

    # Check if image file exists
    if os.path.exists(image_file):
        # Load mask and image as NumPy arrays
        mask = cv2.imread(mask_file,cv2.IMREAD_UNCHANGED)

        # Calculate the scaling factors
        #scale_x = 256 / mask.shape[1]
        #scale_y = 256 / mask.shape[0]

        # Resize the image while maintaining aspect ratio
        #mask = cv2.resize(mask, None, fx=scale_x, fy=scale_y)

        image = cv2.imread(image_file)
        #image = cv2.resize(image, None, fx=scale_x, fy=scale_y)

        binary_image =(np.array(mask) > 128).astype(int)


        # Append mask and image arrays to lists
        mask_list.append(binary_image)
        image_list.append(image)

# Convert lists to NumPy arrays
mask_list = np.array(mask_list)
image_list = np.array(image_list)

# Print the shapes of arrays
print("Shape of Masks Array:", mask_list.shape)
print("Shape of Images Array:", image_list.shape)

with open("print2.txt", mode='a', newline="") as file:
    file.write("Shape of Masks Array:"+ str(mask_list.shape))
    file.write("Shape of Images Array:"+str(image_list.shape))

# images.shape
# image_list.shape


images = image_list
masks = mask_list

images.shape

# Create a list to store the indices of non-empty masks
valid_indices = [i for i, mask in enumerate(masks) if mask.max() != 0]
# Filter the image and mask arrays to keep only the non-empty pairs
filtered_images = images[valid_indices]
filtered_masks = masks[valid_indices]
print("Image shape:", filtered_images.shape)  # e.g., (num_frames, height, width, num_channels)
print("Mask shape:", filtered_masks.shape)

from datasets import Dataset
from PIL import Image

dataset_dict = {
    "image": [Image.fromarray(img) for img in filtered_images],
    "label": [Image.fromarray((mask * 255).astype(np.uint8)) for mask in filtered_masks],
}

# Create the dataset using the datasets.Dataset class
dataset = Dataset.from_dict(dataset_dict)

img_num = random.randint(0, filtered_images.shape[0]-1)
example_image = dataset[img_num]["image"]
example_mask = dataset[img_num]["label"]

fig, axes = plt.subplots(1, 2, figsize=(10, 5))

# Hide axis ticks and labels
for ax in axes:
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xticklabels([])
    ax.set_yticklabels([])

# Display the images side by side
# plt.show()

#Get bounding boxes from mask.
def get_bounding_box(ground_truth_map):
  # get bounding box from mask
  y_indices, x_indices = np.where(ground_truth_map > 0)
  x_min, x_max = np.min(x_indices), np.max(x_indices)
  y_min, y_max = np.min(y_indices), np.max(y_indices)
  # add perturbation to bounding box coordinates
  H, W = ground_truth_map.shape
  x_min = max(0, x_min - np.random.randint(0, 20))
  x_max = min(W, x_max + np.random.randint(0, 20))
  y_min = max(0, y_min - np.random.randint(0, 20))
  y_max = min(H, y_max + np.random.randint(0, 20))
  bbox = [x_min, y_min, x_max, y_max]

  return bbox

from torch.utils.data import Dataset

class SAMDataset(Dataset):
  def __init__(self, dataset, processor):
    self.dataset = dataset
    self.processor = processor

  def __len__(self):
    return len(self.dataset)

  def __getitem__(self, idx):
    item = self.dataset[idx]
    image = item["image"]
    ground_truth_mask = np.array(item["label"])

    # get bounding box prompt
    prompt = get_bounding_box(ground_truth_mask)

    # prepare image and prompt for the model
    inputs = self.processor(image, input_boxes=[[prompt]], return_tensors="pt")

    # remove batch dimension which the processor adds by default
    inputs = {k:v.squeeze(0) for k,v in inputs.items()}

    # add ground truth segmentation
    inputs["ground_truth_mask"] = ground_truth_mask

    return inputs

# Initialize the processor
from transformers import SamProcessor
processor = SamProcessor.from_pretrained("facebook/sam-vit-base")

# Create an instance of the SAMDataset
train_dataset = SAMDataset(dataset=dataset, processor=processor)

example = train_dataset[0]
for k,v in example.items():
  print(k,v.shape)

# Create a DataLoader instance for the training dataset
from torch.utils.data import DataLoader
train_dataloader = DataLoader(train_dataset, batch_size=2, shuffle=True, drop_last=False)

batch = next(iter(train_dataloader))
for k,v in batch.items():
  print(k,v.shape)

# batch["ground_truth_mask"].shape

# Load the model
from transformers import SamModel
model = SamModel.from_pretrained("facebook/sam-vit-base")

# make sure we only compute gradients for mask decoder
for name, param in model.named_parameters():
  if name.startswith("vision_encoder") or name.startswith("prompt_encoder"):
    param.requires_grad_(False)

from torch.optim import Adam
# import monai.losses
# Initialize the optimizer and the loss function
optimizer = Adam(model.mask_decoder.parameters(), lr=1e-5, weight_decay=0)

import torch
import torch.nn as nn
import torch.nn.functional as F

class DiceCELoss(nn.Module):
    def __init__(self, smooth=1e-5, squared_pred=True, reduction='mean'):
        super(DiceCELoss, self).__init__()
        self.smooth = smooth
        self.squared_pred = squared_pred
        self.reduction = reduction

    def forward(self, input, target):
        # Apply sigmoid activation
        input = torch.sigmoid(input)

        # Square the predictions if required
        if self.squared_pred:
            input = input * input

        # Clamp target values between 0 and 1
        target = torch.clamp(target, 0, 1)

        # Calculate Dice coefficient
        intersection = torch.sum(input * target)
        dice_coeff = (2. * intersection + self.smooth) / (torch.sum(input) + torch.sum(target) + self.smooth)

        # Calculate Cross Entropy loss
        ce_loss = nn.BCELoss(reduction=self.reduction)(input, target)

        # Combine Dice coefficient and Cross Entropy loss
        dice_ce_loss = 1 - dice_coeff + ce_loss

        if self.reduction == 'mean':
            return torch.mean(dice_ce_loss)
        elif self.reduction == 'sum':
            return torch.sum(dice_ce_loss)
        else:
            return dice_ce_loss
        
def focal_loss(y_true, y_pred, alpha=0.25, gamma=2.0):
    # Sigmoid activation and binary cross-entropy
    cross_entropy = torch.nn.functional.binary_cross_entropy_with_logits(y_pred, y_true)
    probs = torch.sigmoid(y_pred)

    # Compute alpha for balancing class weights
    alpha_tensor = torch.where(y_true == 1, torch.tensor(alpha), torch.tensor(1.0 - alpha))

    # Compute focal loss
    pt = torch.where(y_true == 1, probs, 1 - probs)
    loss = alpha_tensor * torch.pow(1 - pt, gamma) * cross_entropy

    return torch.mean(loss)

# Replace seg_loss with DiceCELoss
seg_loss = focal_loss #DiceCELoss(smooth=1e-5, squared_pred=True, reduction='mean')


# seg_loss = monai.losses.DiceFocalLoss(sigmoid=True, squared_pred=True, reduction='mean')

from tqdm import tqdm
from statistics import mean
import torch
from torch.nn.functional import threshold, normalize

#Training loop
num_epochs = 20

device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

model.train()
for epoch in range(num_epochs):
    epoch_losses = []
    for batch in tqdm(train_dataloader):
      # forward pass
      outputs = model(pixel_values=batch["pixel_values"].to(device),
                      input_boxes=batch["input_boxes"].to(device),
                      multimask_output=False)

      # compute loss
      predicted_masks = outputs.pred_masks.squeeze(1)
      ground_truth_masks = batch["ground_truth_mask"].float().to(device)
      loss = seg_loss(predicted_masks, ground_truth_masks.unsqueeze(1))
      with open("print2.txt", mode = 'a', newline = "") as file:
        file.write(f"Inner loss: {loss}\n")

      # backward pass (compute gradients of parameters w.r.t. loss)
      optimizer.zero_grad()
      loss.backward()

      # optimize
      optimizer.step()
      epoch_losses.append(loss.item())

    print(f'EPOCH: {epoch}')
    print(f'Mean loss: {mean(epoch_losses)}')
    with open("print2.txt", mode = 'a', newline = "") as file:
        file.write(f"Epoch {epoch + 1}: Mean loss: {mean(epoch_losses)}\n")

# Save the model's state dictionary to a file
torch.save(model.state_dict(), "./models/model_checkpoint.pth")




"""**Testing**"""

from transformers import SamModel, SamConfig, SamProcessor
import torch

# Load the model configuration
model_config = SamConfig.from_pretrained("facebook/sam-vit-base")
processor = SamProcessor.from_pretrained("facebook/sam-vit-base")

# Create an instance of the model architecture with the loaded configuration
my_hist_model = SamModel(config=model_config)
#Update the model by loading the weights from saved file.
my_hist_model.load_state_dict(torch.load("./models/model_checkpoint.pth"))

# set the device to cuda if available, otherwise use cpu
device = "cuda" if torch.cuda.is_available() else "cpu"
my_hist_model.to(device)

import numpy as np
import random
import matplotlib.pyplot as plt

# let's take a random training example
idx = random.randint(0, filtered_images.shape[0]-1)

# load image
test_image = dataset[idx]["image"]

# get box prompt based on ground truth segmentation map
ground_truth_mask = np.array(dataset[idx]["label"])
prompt = get_bounding_box(ground_truth_mask)

# prepare image + box prompt for the model
inputs = processor(test_image, input_boxes=[[prompt]], return_tensors="pt")

# Move the input tensor to the GPU if it's not already there
inputs = {k: v.to(device) for k, v in inputs.items()}

my_hist_model.eval()

# forward pass
with torch.no_grad():
    outputs = my_hist_model(**inputs, multimask_output=False)

# apply sigmoid
medsam_seg_prob = torch.sigmoid(outputs.pred_masks.squeeze(1))
# convert soft mask to hard mask
medsam_seg_prob = medsam_seg_prob.cpu().numpy().squeeze()
medsam_seg = (medsam_seg_prob > 0.5).astype(np.uint8)


fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# Plot the first image on the left
axes[0].imshow(np.array(test_image), cmap='gray')  # Assuming the first image is grayscale
axes[0].set_title("Image")

# Plot the second image on the right
axes[1].imshow(medsam_seg, cmap='gray')  # Assuming the second image is grayscale
axes[1].set_title("Mask")

# Plot the second image on the right
axes[2].imshow(medsam_seg_prob)  # Assuming the second image is grayscale
axes[2].set_title("Probability Map")

# Hide axis ticks and labels
for ax in axes:
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xticklabels([])
    ax.set_yticklabels([])

# Display the images side by side
plt.savefig("img.png")
plt.close()