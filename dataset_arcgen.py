from torch.utils.data import Dataset, DataLoader
import cv2
import os
import torch
from pytorch_lightning import LightningDataModule
from torchvision import transforms

class ArcGenDataset(Dataset):
    """ ArcGen dataset.
    Pytorch Dataset class for ARC-GEN dataset. ARC-GEN dataset is a variation of the CATER dataset.
    It is a dataset of videos of synthetic scenes with a single object moving around.
    The dataset is split into 4 parts: train, val, test and test_val.
    The train and val sets are used for training and validation.
    The test set is used for testing the model on unseen combinations of actions, colors, material and objects.
    The test_val set is used for validating the model on unseen combinations of actions, colors, material and objects.

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
        SPLITDATASET (bool): Whether to split each video into 3 parts. Defaults to True.
        train_data (list): List of training videos.
        val_data (list): List of validation videos.
        test_data (list): List of test videos.
        test_val_data (list): List of test_val videos.

    Methods:
        load_data(): Loads the data (videos and labels) from the dataset.
        __len__(): Returns the length of the dataset.
        __getitem__(): Returns the data point at the given index.
        load_video(): Loads the video at the given path.
    """
    def __init__(self, config, mode='train', transform=None):
        self.config = config
        self.stride = config['input_stride']
        self.mode = mode
        self.data_path = config['data_path']
        self.transform = transform
        self.SPLITDATASET = True # it's hardcoded for now. Will split each video into 3

        self.train_data, self.val_data, self.test_data, self.test_val_data = self.load_data()

    def load_data(self):
        """ Loads the data (videos and labels) from the dataset.

        Returns:
            train_data (list): List of training videos.
            val_data (list): List of validation videos.
            test_data (list): List of test videos.
            test_val_data (list): List of test_val videos.
        """
        train_data = []
        val_data = []
        test_data = []
        test_val_data = []

        with open(os.path.join(self.data_path, 'train.txt'), 'r') as f:
            for line in f:
                train_data.append(line.strip())

        with open(os.path.join(self.data_path, 'val.txt'), 'r') as f:
            for line in f:
                val_data.append(line.strip())

        with open(os.path.join(self.data_path, 'test.txt'), 'r') as f:
            for line in f:
                test_data.append(line.strip())
        
        with open(os.path.join(self.data_path, 'test_val.txt'), 'r') as f:
            for line in f:
                test_val_data.append(line.strip())

        return train_data, val_data, test_data, test_val_data

    def __len__(self):
        """ Returns the length of the dataset.

        Returns:
            l (int): Length of the dataset.
        """
        if self.config['debug']:
            l = 10
        elif self.mode == 'train':
            l = len(self.train_data)
        elif self.mode == 'val':
            l = len(self.val_data)
        elif self.mode == 'test_val':
            l = len(self.test_val_data)
        else:
            l = len(self.test_data)

        if self.SPLITDATASET:
            l = l * 3
        return l

    def __getitem__(self, idx):
        """ Returns the data point at the given index.

        Args:
            idx (int): Index of the data point.

        Returns:
            frames (torch.Tensor): Tensor of shape (3, 224, 224, 30) containing the input images.
            label (torch.Tensor): Tensor of shape (sentence_length) containing the labels.
        """
        if self.SPLITDATASET:
            new_idx = idx // 3
        else:
            new_idx = idx
        if self.mode == 'train':
            data_point = self.train_data[new_idx]
        elif self.mode == 'val':
            data_point = self.val_data[new_idx]
        elif self.mode == 'test_val':
            data_point = self.test_val_data[new_idx]
        else:
            data_point = self.test_data[new_idx]

        video_path = data_point.split(':')[0]
        video_path = os.path.join(self.data_path, 'images', video_path)

        frames = self.load_video(video_path, 30*idx%3)

        label = data_point.split(':')[1].split(',')
        label = [int(x) for x in label]
        if self.SPLITDATASET:
            label = label[idx%3:idx%3+self.config["sentence_length"]]
        label = torch.tensor(label, dtype=torch.uint8)

        return frames, label

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
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if self.SPLITDATASET and i < idx:
                i += 1
                continue
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

class DataModule(LightningDataModule):
    """ DataModule for the ArcGen dataset.

    Args:
        config (dict): Configuration dictionary.

    Attributes:
        config (dict): Configuration dictionary.
        transform (torchvision.transforms.Normalize): Normalization transform.
        train_dataset (ArcGenDataset): Training dataset.
        val_dataset (ArcGenDataset): Validation dataset.
        test_dataset (ArcGenDataset): Test dataset.
        test_val_dataset (ArcGenDataset): Test dataset for validation.

    Methods:
        setup: Sets up the datasets.
        train_dataloader: Returns the training dataloader.
        val_dataloader: Returns both the validation and test-validation dataloaders.
        test_dataloader: Returns the test dataloader.
    """
    def __init__(self, config):
        super().__init__()
        self.config = config
        mean = torch.tensor([0.4569, 0.4648, 0.4688])
        std = torch.tensor([0.0166, 0.0153, 0.0153])
        self.transform = transforms.Normalize(mean, std)

    def setup(self, stage=None):
        self.train_dataset = ArcGenDataset(self.config, mode='train', transform=self.transform)
        self.val_dataset = ArcGenDataset(self.config, mode='val', transform=self.transform)
        self.test_dataset = ArcGenDataset(self.config, mode='test', transform=self.transform)
        self.test_val_dataset = ArcGenDataset(self.config, mode='test_val', transform=self.transform)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.config['batch_size'], shuffle=True, num_workers=self.config["num_workers"])

    def val_dataloader(self):
        val_loader = DataLoader(self.val_dataset, batch_size=self.config['batch_size'], shuffle=False, num_workers=self.config["num_workers"])
        test_val_loader = DataLoader(self.test_val_dataset, batch_size=self.config['batch_size'], shuffle=False, num_workers=self.config["num_workers"])
        return [val_loader, test_val_loader]

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.config['batch_size'], shuffle=False, num_workers=self.config["num_workers"])