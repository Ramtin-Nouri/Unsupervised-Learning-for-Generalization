from torch.utils.data import Dataset, DataLoader
import cv2
import os
import torch
from pytorch_lightning import LightningDataModule
from torchvision import transforms

class ArcGenDataset(Dataset):
    def __init__(self, config, mode='train', transform=None):
        self.config = config
        self.stride = config['input_stride']
        self.mode = mode
        self.data_path = config['data_path']
        self.transform = transform

        self.train_data, self.val_data, self.test_data, self.test_val_data = self.load_data()

    def load_data(self):
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
        if self.config['debug']:
            return 10
        if self.mode == 'train':
            return len(self.train_data)
        elif self.mode == 'val':
            return len(self.val_data)
        elif self.mode == 'test_val':
            return len(self.test_val_data)
        else:
            return len(self.test_data)

    def __getitem__(self, idx):
        if self.mode == 'train':
            data_point = self.train_data[idx]
        elif self.mode == 'val':
            data_point = self.val_data[idx]
        elif self.mode == 'test_val':
            data_point = self.test_val_data[idx]
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
        i = 0
        while True:
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

        frames = torch.stack(frames)
        frames = frames.permute(0, 3, 1, 2)
        frames = frames.float() / 255.0
        if self.transform is not None:
            frames = self.transform(frames)
        else:
            frames = frames - 0.5
        return frames

class DataModule(LightningDataModule):
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