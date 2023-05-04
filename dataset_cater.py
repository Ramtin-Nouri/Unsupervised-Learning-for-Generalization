from torch.utils.data import Dataset, DataLoader
import cv2
import glob
import os
import torch
from pytorch_lightning import LightningDataModule
from torchvision import transforms

class CaterDataset(Dataset):
    """ Cater dataset for unsupervised learning.
    Pytorch Dataset class for the Cater dataset.
    It is a dataset of videos of synthetic scenes with upto 2 objects moving around.

    Args:
        config (dict): Configuration dictionary.
        mode (str, optional): Mode of the dataset. Can be 'train', 'val', 'test' or 'test_val'. Defaults to 'train'.
        transform (torchvision.transforms, optional): Transformations to apply to the images. Defaults to None.

    Attributes:
        config (dict): Configuration dictionary.
        stride (int): Stride of the input images.
        mode (str): Mode of the dataset. Can be 'train', 'val', 'test' or 'test_val'.
        data_path (str): Path to the dataset.
        transform (torchvision.transforms): Transformations to apply to the images.
        data (list): List of videos.

    Methods:
        load_data(): Loads the data (videos and labels) from the dataset.
        __len__(): Returns the length of the dataset.
        __getitem__(): Returns the data point at the given index.
        load_video(): Loads the video at the given path.
    """
    def __init__(self, config, mode, transform=None):
        self.config = config
        self.stride = config['input_stride']
        self.mode = mode
        self.data_path = config['data_path']
        self.transform = transform

        self.data = self.load_data()

    def load_data(self):
        """ Loads the data (videos) from the dataset.

        Returns:
            data_paths (list): List of videos.
        """    
        data_paths = glob.glob(self.data_path + '/*/videos/*.avi')
        data_paths.sort()
        if self.mode == 'train':
            data_paths = data_paths[:int(0.8*len(data_paths))]
        elif self.mode == 'val':
            data_paths = data_paths[int(0.8*len(data_paths)):]
        elif self.mode == 'test':
            # Use 2 videos of val set for testing
            data_paths = data_paths[int(0.8*len(data_paths)):]
        else:
            raise ValueError(f'Invalid mode {self.mode}')

        return data_paths
        

    def __len__(self):
        """ Returns the length of the dataset.

        Returns:
            l (int): Length of the dataset.
        """
        if self.config['debug']:
            return 10
        if self.mode == 'test':
            return 2
        return len(self.data)

    def __getitem__(self, idx):
        """ Returns the data point at the given index.

        Args:
            idx (int): Index of the data point.

        Returns:
            frames (torch.Tensor): Tensor of shape (3, 224, 224, 30) containing the input images.
            label (torch.Tensor): Tensor of shape (sentence_length) containing the labels.
        """
        video_path = self.data[idx]
        frames = self.load_video(video_path, idx)

        return frames, torch.tensor([]), torch.tensor([]) # for compatibility reasons

    def load_video(self, video_path, idx):
        """ Loads the video at the given path.
        Uses OpenCV to read the video and returns a tensor of shape (sequence_length, channels, height, width).
        Does not use torchvision's VideoReader because it does not work!?
        Does not use hardware acceleration as of now.

        Use input_stride to control the number of frames to skip.
        if SPLITDATASET is True, it will skip frames until it reaches the correct index of the split video.

        Args:
            video_path (str): Path to the video.

        Returns:
            frames (torch.Tensor): Tensor of shape (sequence_length, channels, height, width) containing the input images.
        """
        cap = cv2.VideoCapture(video_path)
        assert cap.isOpened(), f'Cannot capture source {video_path}'
        frames = []
        i = 0
        for _ in range(self.config['input_length']):
            ret, frame = cap.read()
            if not ret:
                break
            if i % self.stride != 0:
                i += 1
                continue
            frame = torch.tensor(frame)
            frames.append(frame)
            i += 1
        cap.release()

        #print(len(frames))
        # TODO: cap to fix length + pad preceding zeros

        frames = torch.stack(frames) # (sequence_length, height, width, channels)
        frames = frames.permute(0, 3, 1, 2) # (sequence_length, channels, height, width)
        frames = frames.float() / 255.0
        if self.transform is not None:
            frames = self.transform(frames)
        else:
            frames = frames - 0.5
        return frames

class CaterModule(LightningDataModule):
    """ DataModule for the Cater dataset.

    Args:
        config (dict): Configuration dictionary.

    Attributes:
        config (dict): Configuration dictionary.
        transform (torchvision.transforms.Normalize): Normalization transform.
        

    Methods:
        setup: Sets up the datasets.
        train_dataloader: Returns the training dataloader.
        val_dataloader: Returns the validation dataloader.
        test_dataloader: Returns the test dataloader.
    """
    def __init__(self, config):
        super().__init__()
        self.config = config
        mean = torch.tensor([0.4569, 0.4648, 0.4688])
        std = torch.tensor([0.0166, 0.0153, 0.0153])
        self.transform = transforms.Normalize(mean, std)

    def setup(self, stage=None):
        """ Sets up the datasets."""
        self.train_dataset = CaterDataset(self.config, mode='train', transform=self.transform)
        self.val_dataset = CaterDataset(self.config, mode='val', transform=self.transform)
        self.test_dataset = CaterDataset(self.config, mode='test', transform=self.transform)

    def train_dataloader(self):
        """ Returns the training dataloader.
        
        Returns:
            DataLoader: Training dataloader.
        """
        return DataLoader(self.train_dataset, batch_size=self.config['batch_size'], shuffle=True, num_workers=self.config["num_workers"])

    def val_dataloader(self):
        """ Returns the validation dataloader.

        Returns:
            DataLoader: Validation dataloader.
        """
        val_loader = DataLoader(self.val_dataset, batch_size=self.config['batch_size'], shuffle=False, num_workers=self.config["num_workers"])
        return val_loader

    def test_dataloader(self):
        """ Returns the test dataloader.

        Returns:
            DataLoader: Test dataloader.
        """
        test_loader = DataLoader(self.test_dataset, batch_size=self.config['batch_size'], shuffle=False, num_workers=self.config["num_workers"])
        return test_loader
