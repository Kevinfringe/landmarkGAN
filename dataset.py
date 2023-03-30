import os
import random
import pandas as pd

import torch
import torchvision.transforms as transforms
from PIL import Image

exp_dict = {
    "AN": 0,
    "DI": 1,
    "FE": 2,
    "HA": 3,
    "NE": 4,
    "SA": 5,
    "SU": 6
}


class jaffeDataset(torch.utils.data.Dataset):
    def __init__(self, train_set_path, train_set_final_path, lm_img_path, lm_csv_path, transform=None):
        self.transform = transform
        self.imgs = []
        self.lm_img_path = lm_img_path
        self.ori_exp_index = 0
        self.target_exp_index = 0

        # loop through the files in train_set_final_path and generate a list of tuples (final_path, original_path)
        for file_name in os.listdir(train_set_final_path):
            if file_name.endswith(".jpg"):
                # extract the prefix, expression type, and index from the file name
                parts = file_name.split(".")
                prefix = parts[0]
                exp_type = parts[2]
                origin_exp_type = parts[1]
                index = parts[1][2:]
                suffix = parts[3]

                self.ori_exp_index = exp_dict[origin_exp_type[:2]]
                self.target_exp_index = exp_dict[exp_type]

                if (int)(index) >= 4:
                    index = 3
                # generate the expected file name in the original train set directory
                original_prefix = f"{prefix}.{exp_type}{index}"
                blur_prefix = f"{prefix}.{exp_type}"
                original_file_name = self.find_original_file(original_prefix, blur_prefix, train_set_path)
                if original_file_name is not None:
                    original_path = os.path.join(train_set_path, original_file_name)
                    final_path = os.path.join(train_set_final_path, file_name)
                    lm_path = os.path.join(lm_img_path, f"{prefix}.{origin_exp_type}.{suffix}.jpg")
                    lm_csv = os.path.join(lm_csv_path, f"{prefix}.{origin_exp_type}.{suffix}.csv")
                    self.imgs.append((final_path, original_path, lm_path, lm_csv))

    def __getitem__(self, index):
        final_path, original_path, lm_path, lm_csv = self.imgs[index]

        target_img = Image.open(original_path).convert('RGB')
        lm_img = Image.open(lm_path).convert('RGB')
        img = Image.open(final_path).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)
            lm_img = self.transform(lm_img)
            target_img = self.transform(target_img)

        # Get target landmark coordinate
        df = pd.read_csv(lm_csv)
        lm_array = df.to_numpy()
        lm_array = torch.tensor(lm_array)

        # define landmark generator label.
        lm_gen_label = torch.zeros((7,))
        lm_gen_label[self.ori_exp_index] = 1
        exp_gen_label = torch.zeros((7,))
        exp_gen_label[self.ori_exp_index] = 1

        return img, lm_img, target_img, lm_array, lm_gen_label, exp_gen_label

    def __len__(self):
        return len(self.imgs)

    def find_original_file(self, original_prefix, blur_prefix, train_set_path):
        matching_files = []
        for file_name in os.listdir(train_set_path):
            if file_name.startswith(original_prefix):
                return file_name
            elif file_name.startswith(blur_prefix):
                matching_files.append(file_name)
        if len(matching_files) == 0:
            # If we get to this point, no matching file was found
            print(f"No matching file found for prefix {original_prefix} in {train_set_path}. Skipping file...")
            return None
        elif len(matching_files) == 1:
            # If there is only one matching file, return it
            return matching_files[0]
        else:
            # If there are multiple matching files, randomly select one
            matching_files = [f for f in matching_files if not f.endswith(".gif")]
            if len(matching_files) == 0:
                # If there are no non-gif files, use all files
                matching_files = [f for f in matching_files if f.endswith(".jpg") or f.endswith(".png")]
            return random.choice(matching_files)


import matplotlib.pyplot as plt

# # Below code is only for testing.
# set the paths to the training and validation sets, as well as the landmarks images
# train_set_path = "./jaffedbase_official/train_set_jpg/"
# train_set_final_path = "./jaffedbase_official/train_set_final/"
# lm_img_path = "./jaffedbase_official/lm_img/"
# lm_csv_path = "./jaffedbase_official/lm_csv/"
#
# # define the image transformations
# # define the image transformations
# transform = transforms.Compose([
#     transforms.Resize((256, 256)),
#     transforms.RandomHorizontalFlip(),
#     transforms.ToTensor(),
# ])
#
# # create the jaffeDataset object
# dataset = jaffeDataset(train_set_path, train_set_final_path, lm_img_path,lm_csv_path, transform=transform)
#
# # create the data loader for the dataset
# data_loader = torch.utils.data.DataLoader(dataset, batch_size=4, shuffle=True)
#
# # loop through the data loader and plot the images
# for img, lm_img, target_img, lm_array in data_loader:
#     print(img.size())
#     # convert the tensor images to numpy arrays and transpose the dimensions
#     img = img.numpy().transpose((0, 2, 3, 1))
#     lm_img = lm_img.numpy().transpose((0, 2, 3, 1))
#     target_img = target_img.numpy().transpose((0, 2, 3, 1))
#     # create a figure with subplots for each image in the batch
#     fig, axes = plt.subplots(nrows=4, ncols=3, figsize=(10, 10))
#     for i in range(4):
#         # plot the original image
#         axes[i][0].imshow(img[i])
#         axes[i][0].set_title("Original Image")
#         # plot the landmarks image
#         axes[i][1].imshow(lm_img[i])
#         axes[i][1].set_title("Landmarks Image")
#         # plot the target image
#         axes[i][2].imshow(target_img[i])
#         axes[i][2].set_title("Target Image")
#     plt.show()
#     break

