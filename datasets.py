import numpy as np
import cv2
import random
import torch
from torch.utils.data import Dataset
from torchvision import transforms, utils
from loguru import logger
from albumentations.pytorch import ToTensorV2
import albumentations as A
import torchvision.transforms.functional as F

import warnings
warnings.filterwarnings("ignore")


def center_crop(img, dim):
    """
    Center crop an image.
    This function crops an image by taking a rectangular region from the center of the image.
    The size of the rectangular region is determined by the input dimensions.
    Args:
        img (ndarray): The image to be cropped. The image should be a 2D numpy array.
        dim (tuple): A tuple of integers representing the width and height of the crop window.
    Returns:
        ndarray: The center-cropped image as a 2D numpy array.
    Example:
        >>> import numpy as np
        >>> img = np.random.randint(0, 255, size=(100, 100, 3), dtype=np.uint8)
        >>> cropped_img = center_crop(img, (50, 50))
    """
    width, height = img.shape[1], img.shape[0]
    # process crop width and height for max available dimension
    crop_width = dim[0] if dim[0] < img.shape[1] else img.shape[1]
    crop_height = dim[1] if dim[1] < img.shape[0] else img.shape[0]
    mid_x, mid_y = int(width / 2), int(height / 2)
    cw2, ch2 = int(crop_width / 2), int(crop_height / 2)
    crop_img = img[mid_y - ch2:mid_y + ch2, mid_x - cw2:mid_x + cw2]
    return np.ascontiguousarray(crop_img)


def get_random_crop(image, mask, crop_width=256, crop_height=256):
    """
    Get a random crop from the image and mask.

    Args:
        image (numpy.ndarray): The input image (numpy array).
        mask (numpy.ndarray): The input mask (numpy array).
        crop_width (int): Width of the crop (default is 256).
        crop_height (int): Height of the crop (default is 256).

    Returns:
        tuple: A tuple containing the cropped image and mask as numpy arrays.
    """
    max_x = mask.shape[1] - crop_width
    max_y = mask.shape[0] - crop_height
    x = np.random.randint(0, max_x)
    y = np.random.randint(0, max_y)
    crop_mask = mask[y: y + crop_height, x: x + crop_width]
    crop_image = image[y: y + crop_height, x: x + crop_width, :]
    return crop_image, crop_mask


def data_augmentation(x, y):


    apply_transform = ["yes", "no"]

    if random.choice(apply_transform)=="yes":

        # Define your augmentation transformations
        transform = A.Compose([
            A.HorizontalFlip(p=0.5),  # Random horizontal flip with a probability of 0.5
            A.VerticalFlip(p=0.5),    # Random vertical flip with a probability of 0.5

        ])

        # Perform augmentation on the image
        x_augmented = transform(image=x)
        y_augmented = transform(image=y)
        # Retrieve the augmented image
        x_augmented = x_augmented['image']
        y_augmented = y_augmented['image']


    else:
        x_augmented = x
        y_augmented = y

    return x_augmented, y_augmented


def normalize(band):
    band_min, band_max = (band.min(), band.max())
    return ((band-band_min)/((band_max - band_min)))


def image_preprocessing(image_path):

    image = xr.open_rasterio(image_path, masked=False).values
    red = image[3,:,:]
    green = image[2,:,:]
    blue = image[1,:,:]
    red_n = normalize(red)
    green_n = normalize(green)
    blue_n = normalize(blue)
    rgb_composite_n= np.dstack((red_n, green_n, blue_n))
    return rgb_composite_n


def image_preprocessing_index(image_path):

    image = xr.open_rasterio(image_path, masked=False).values

    nwdi = (image[2,:,:]-image[7,:,:])/(image[2,:,:]+image[7,:,:])
    # nwdi = normalize(nwdi)

    ndvi = (image[7,:,:]-image[3,:,:])/(image[7,:,:]+image[3,:,:])
    # ndvi = normalize(ndvi)

    msi = image[10,:,:]/image[7,:,:]
    # msi = normalize(msi)

    image_index = np.dstack((ndvi, nwdi, msi))

    image_index = normalize(image_index)
    print(image_index.shape)
    # image_index= np.transpose(image_index, (1, 2, 0))
    return image_index


def image_preprocessing_pca(image_path):


    from sklearn.decomposition import PCA

    image = xr.open_rasterio(image_path, masked=False).values
    reshaped_data = image.reshape((12, -1)).T  # Transpose for the correct shape
    n_components = 3  # Number of principal components
    pca = PCA(n_components=n_components)
    pca_result = pca.fit_transform(reshaped_data)

    pca_result_reshaped = pca_result.T.reshape((n_components, 512, 512))
    # pca_result_reshaped = normalize(pca_result_reshaped)
    pca_result_reshaped = np.transpose(pca_result_reshaped, (1, 2, 0))
    return pca_result_reshaped


class TrainDataset(Dataset):
    """
    Custom training dataset class.
    """

    def __init__(self, ds_inputs, ds_target):

        self.ds_inputs = ds_inputs
        self.ds_target = ds_target


    def __getitem__(self, index):

        x = self.ds_inputs[index].values

        y = self.ds_target[index].values


        y_norm = (y-y.min())/(y.max() - y.min()) + 0.01

        x_norm = (x-y.min())/(y.max() - y.min()) + 0.01

        x_norm[np.isnan(x_norm)] = 0.001

        transform = transforms.Compose([
            transforms.ToTensor()])

        x_norm = transform(x_norm)

        y_norm = torch.tensor(y_norm, dtype=torch.float32).unsqueeze(0)

        return x_norm, y_norm

    def __len__(self):
        return len(self.ds_inputs)


class EvalDataset(Dataset):

    """
    Custom training dataset class.
    """

    def __init__(self, ds_inputs, ds_target):

        self.ds_inputs = ds_inputs
        self.ds_target = ds_target


    def __getitem__(self, index):

        x = self.ds_inputs[index].values

        y = self.ds_target[index].values


        y_norm = (y-y.min())/(y.max() - y.min()) + 0.01

        x_norm = (x-y.min())/(y.max() - y.min()) + 0.01

        x_norm[np.isnan(x_norm)] = 0.001



        # x = cv2.merge((x, x, x))
        # x , y = data_augmentation(x,y)

        transform = transforms.Compose([
            transforms.ToTensor()])

        x_norm = transform(x_norm)

        y_norm = torch.tensor(y_norm, dtype=torch.float32).unsqueeze(0)

        return x_norm, y_norm

    def __len__(self):
        return len(self.ds_inputs)


class TestDataset(Dataset):

    """
    Custom training dataset class.
    """

    def __init__(self, ds_inputs, ds_target):

        self.ds_inputs = ds_inputs
        self.ds_target = ds_target


    def __getitem__(self, index):

        x = self.ds_inputs[index].values

        y = self.ds_target[index].values


        y_norm = (y-y.min())/(y.max() - y.min()) + 0.01

        x_norm = (x-y.min())/(y.max() - y.min()) + 0.01

        x_norm[np.isnan(x_norm)] = 0.001

        transform = transforms.Compose([
            transforms.ToTensor()])

        x_norm = transform(x_norm)

        y_norm = torch.tensor(y_norm, dtype=torch.float32).unsqueeze(0)

        return x_norm, y_norm

    def __len__(self):
        return len(self.ds_inputs)
