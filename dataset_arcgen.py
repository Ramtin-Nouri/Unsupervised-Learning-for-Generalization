from torch.utils.data import Dataset, DataLoader
import cv2
import os
import torch
from pytorch_lightning import LightningDataModule

class ArcGenDataset(Dataset):
    def __init__(self, config, mode='train'):
        self.config = config
        self.mode = mode
        self.data_path = config['data_path']

        self.train_data, self.val_data, self.test_data = self.load_data()

    def load_data(self):
        train_data = []
        val_data = []
        test_data = []

        with open(os.path.join(self.data_path, 'train.txt'), 'r') as f:
            for line in f:
                train_data.append(line.strip())

        with open(os.path.join(self.data_path, 'val.txt'), 'r') as f:
            for line in f:
                val_data.append(line.strip())

        with open(os.path.join(self.data_path, 'test.txt'), 'r') as f:
            for line in f:
                test_data.append(line.strip())

        return train_data, val_data, test_data

    def __len__(self):
        if self.config['debug']:
            return 10
        if self.mode == 'train':
            return len(self.train_data)
        elif self.mode == 'val':
            return len(self.val_data)
        else:
            return len(self.test_data)

    def __getitem__(self, idx):
        if self.mode == 'train':
            data_point = self.train_data[idx]
        elif self.mode == 'val':
            data_point = self.val_data[idx]
        else:
            data_point = self.test_data[idx]

        video_path = data_point.split(':')[0]
        video_path = os.path.join(self.data_path, 'images', video_path)

        frames = self.load_video(video_path)

        label = data_point.split(':')[1].split(',')
        label = [int(x) for x in label]
        label = torch.tensor(label, dtype=torch.uint8)

        return frames, label

    def load_video(self, video_path):
        cap = cv2.VideoCapture(video_path)
        assert cap.isOpened(), f'Cannot capture source {video_path}'
        frames = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame = torch.tensor(frame)
            frames.append(frame)
        cap.release()

        #print(len(frames))
        # TODO: cap to fix length + pad preceding zeros

        frames = torch.stack(frames)
        frames = frames.permute(0, 3, 1, 2)
        frames = frames.float() / 255.0
        frames = frames - 0.5 # TODO: do proper normalization
        return frames

class DataModule(LightningDataModule):
    def __init__(self, config):
        super().__init__()
        self.config = config
        # TODO: add transforms

    def setup(self, stage=None):
        self.train_dataset = ArcGenDataset(self.config, mode='train')
        self.val_dataset = ArcGenDataset(self.config, mode='val')
        self.test_dataset = ArcGenDataset(self.config, mode='test')

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.config['batch_size'], shuffle=True, num_workers=self.config["num_workers"])

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.config['batch_size'], shuffle=False, num_workers=self.config["num_workers"])

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.config['batch_size'], shuffle=False, num_workers=self.config["num_workers"])