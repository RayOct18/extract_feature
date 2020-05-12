import pandas as pd
import os
import numpy as np
from PIL import Image
import torch
from torch.utils.data.dataset import Dataset  # For custom datasets
from imgaug import augmenters as iaa


class CustomDatasetFromImages(Dataset):
    def __init__(self, dataset_path=None, csv_path=None, transform=None):
        """
        Args:
            csv_path (string): path to csv file
            img_path (string): path to the folder where images are
            transform: pytorch transforms for transforms and tensor conversion
        """
        # Transforms
        self.transform = transform
        # Read the csv file
        self.dataset_path = dataset_path
        self.data_info = pd.read_csv(csv_path, header=0)
        # First column contains the image paths
        self.image_arr = np.asarray(self.data_info.iloc[:, 0])
        # Second column is the labels
        self.label_arr = np.asarray(self.data_info.iloc[:, 1])
        # self.gender_arr = np.asarray(self.data_info.iloc[:, 2])
        # Calculate len
        self.data_len = len(self.data_info.index)
            

    def __getitem__(self, index):
        # Get image name from the pandas df
        single_image_name = self.image_arr[index]
        # Open image
        img_as_img = Image.open(os.path.join(self.dataset_path, *single_image_name.split('/')))
        # img_as_img = img_as_img.convert('RGB')
        if self.transform is not None:
            img_as_img = self.transform(img_as_img)

        # Get label(class) of the image based on the cropped pandas column
        age_label = self.label_arr[index]
        # gender_label = self.gender_arr[index]

        return (img_as_img, age_label, single_image_name)

    def __len__(self):
        return self.data_len
