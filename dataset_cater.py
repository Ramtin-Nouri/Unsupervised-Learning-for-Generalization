from torch.utils.data import Dataset, DataLoader
from pytorch_lightning import LightningDataModule
import torch
import cv2
import os

class CaterDataset(Dataset):
    """PyTorch Dataset for CATER dataset
    
    Reads in and provides the data to PyTorch DataLoader.
    Each data point is a tuple of (image_sequence, label).
    Each image_sequence consists of exactly 301 images.

    Args:
        data_path (str): Path to the data directory.
        split (str): Which split to use. One of 'train', 'val'.
        task (str): Which task to use. One of 'actions_present' , 'actions_order_uniq'.
        dictionary_size (int): Size of the dictionary to use.
        transform (callable): Transformation to apply to each image.
        debug (bool): Whether to use a small subset of the data for debugging.
    """
    def __init__(self, data_path, split, task, dictionary_size, transform=None,debug=False):
        self.data_path = data_path
        self.split = split
        assert self.split in ['train', 'val']

        self.task = task
        assert self.task in ['actions_present', 'actions_order_uniq']

        self.dictionary_size = dictionary_size
        self.transform = transform
        self.debug = debug
        self.data_files = self._load_data()
    
    def __len__(self):
        """Returns the number of data points in the dataset."""
        if self.debug:
            return 10
        return len(self.data_files)

    def __getitem__(self, idx):
        """Returns the idx-th data point in the dataset."""
        data_file, label = self.data_files[idx]
        image_sequence = self._load_video(data_file)
        if self.transform:
            image_sequence = self.transform(image_sequence)
        return image_sequence, label

    def _load_data(self):
        """Loads the data files and labels."""
        data_files = []
        lines = open(os.path.join(self.data_path, "lists", self.task, self.split + '.txt'), 'r').readlines()
        for line in lines:
            data_file, label = line.split(' ')
            data_file = os.path.join(self.data_path, "videos", data_file)
            label = label.split(',')
            label = [int(l) for l in label]
            label_tensor = torch.zeros(self.dictionary_size)
            label_tensor[label] = 1 # one-hot encoding
            data_files.append((data_file, label_tensor))

        data_files.sort()
        return data_files

    def _load_video(self, data_file):
        # Read in the video
        # TODO: use hardware acceleration
        cap = cv2.VideoCapture(data_file)
        assert cap.isOpened()
        image_sequence = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame = torch.from_numpy(frame).float()/255.0
            frame = frame.permute(2, 0, 1)

            image_sequence.append(frame)
        cap.release()
        assert len(image_sequence) == 301
        image_sequence = torch.stack(image_sequence, dim=0)
        return image_sequence

class DataModule(LightningDataModule):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.data_path = config["data_path"]
        self.batch_size = config["batch_size"]
        self.num_workers = config["num_workers"]
        self.task = config["task"]
        self.dictionary_size = config["dictionary_size"]
        self.debug = config["debug"]
        
        self.transform = None
        # TODO: add transform Normalize

    def setup(self, stage=None):
        self.train_dataset = CaterDataset(self.data_path, 'train', self.task, self.dictionary_size,self.transform, debug=self.debug)
        self.val_dataset = CaterDataset(self.data_path, 'val', self.task, self.dictionary_size, self.transform, debug=self.debug)
        if not self.debug:
            assert len(self.train_dataset) + len(self.val_dataset) == 5500

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)