import random
import warnings

import albumentations as A  # type: ignore
import numpy as np  # type: ignore
import torch  # type: ignore
from torch.utils.data import Dataset  # type: ignore
from torchvision import transforms  # type: ignore

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


def create_3d_datasets(ds_inputs, ds_target, depth=6):
    data_inputs = ds_inputs.values
    data_target = ds_target.values

    new_inputs = []
    new_target = []
    for i in range(0, len(data_inputs)):

        if i < (depth/2):

            input_slice = data_inputs[0: depth, :, :]

        elif i > (depth/2) and i < (len(data_inputs)- (depth/2)):

            input_slice = data_inputs[i - (int(depth/2)): i + (int(depth/2)), :, :]
        else:
            input_slice = data_inputs[- depth:, :, :]
        target_slice = data_target[i, :, :]
        new_inputs.append(input_slice)
        new_target.append(target_slice)

    new_inputs = np.array(new_inputs)
    new_target = np.array(new_target)

    return new_inputs, new_target


class TrainDataset(Dataset):
    """
    Custom training dataset class.
    """

    def __init__(self, ds_inputs, ds_target):

        self.ds_inputs = ds_inputs
        self.ds_target = ds_target


    def __getitem__(self, index):

        x = self.ds_inputs[index].values
        # x = cv2.merge((x, x, x))


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
        # x = cv2.merge((x, x, x))


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


class TestDataset(Dataset):

    """
    Custom training dataset class.
    """

    def __init__(self, ds_inputs, ds_target):

        self.ds_inputs = ds_inputs
        self.ds_target = ds_target


    def __getitem__(self, index):

        x = self.ds_inputs[index].values
        # x = cv2.merge((x, x, x))


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


class TrainDataset3D(Dataset):
    def __init__(self, ds_inputs, ds_target):
        self.ds_inputs = ds_inputs
        self.ds_target = ds_target

    def __getitem__(self, index):
        x = self.ds_inputs[index]  # 3D numpy array (depth, height, width)
        y = self.ds_target[index]  # 3D numpy array (depth, height, width)

        # Normalize the data
        y_min, y_max = y.min(), y.max()
        y_norm = (y - y_min) / (y_max - y_min) + 0.01
        x_norm = (x - y_min) / (y_max - y_min) + 0.01
        x_norm[np.isnan(x_norm)] = 0.001

        # Transform to tensors and add channel dimension
        x_norm = torch.tensor(x_norm, dtype=torch.float32).unsqueeze(0)  # Add channel dimension
        y_norm = torch.tensor(y_norm, dtype=torch.float32).unsqueeze(0)  # Add channel dimension

        return x_norm, y_norm

    def __len__(self):
        return len(self.ds_inputs)


class TestDataset3D(Dataset):
    def __init__(self, ds_inputs, ds_target):
        self.ds_inputs = ds_inputs
        self.ds_target = ds_target

    def __getitem__(self, index):
        x = self.ds_inputs[index]  # 3D numpy array (depth, height, width)
        y = self.ds_target[index]  # 3D numpy array (depth, height, width)

        # Normalize the data
        y_min, y_max = y.min(), y.max()
        y_norm = (y - y_min) / (y_max - y_min) + 0.01
        x_norm = (x - y_min) / (y_max - y_min) + 0.01
        x_norm[np.isnan(x_norm)] = 0.001

        # Transform to tensors and add channel dimension
        x_norm = torch.tensor(x_norm, dtype=torch.float32).unsqueeze(0)  # Add channel dimension
        y_norm = torch.tensor(y_norm, dtype=torch.float32).unsqueeze(0)  # Add channel dimension

        return x_norm, y_norm

    def __len__(self):
        return len(self.ds_inputs)


class EvalDataset3D(Dataset):
    def __init__(self, ds_inputs, ds_target):
        self.ds_inputs = ds_inputs
        self.ds_target = ds_target

    def __getitem__(self, index):
        x = self.ds_inputs[index]  # 3D numpy array (depth, height, width)
        y = self.ds_target[index]  # 3D numpy array (depth, height, width)

        # Normalize the data
        y_min, y_max = y.min(), y.max()
        y_norm = (y - y_min) / (y_max - y_min) + 0.01
        x_norm = (x - y_min) / (y_max - y_min) + 0.01
        x_norm[np.isnan(x_norm)] = 0.001

        # Transform to tensors and add channel dimension
        x_norm = torch.tensor(x_norm, dtype=torch.float32).unsqueeze(0)  # Add channel dimension
        y_norm = torch.tensor(y_norm, dtype=torch.float32).unsqueeze(0)  # Add channel dimension

        return x_norm, y_norm

    def __len__(self):
        return len(self.ds_inputs)