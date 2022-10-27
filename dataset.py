import glob
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision.io import read_image


class MultimodalSimulation(Dataset):
    """PyTorch Dataset Implementation.
    
    Reads in and provides the dataset to PyTorch.
    The sequences are loaded from the disk on the fly.
    They may have different lengths.
    The input sequences are padded with the first frame to the input_length.
    If a stride is given, the input sequences are subsampled.
    
    #TODO verfiy comment written by Copilot:
    Args:
        path (str): Path to the dataset.
        part (str): Part of the dataset to use. Can be "training", "validation", "constant-test" or "generalization-test".
        visible_objects (list): List of objects that are visible in the scene. Can be "apple", "banana", "cup", "football", "book", "pylon", "bottle", "star", "ring".
        different_actions (bool): If True, the actions are different for each object. If False, the actions are the same for all objects.
        different_colors (bool): If True, the colors are different for each object. If False, the colors are the same for all objects.
        different_objects (bool): If True, the objects are different for each object. If False, the objects are the same for all objects.
        exclusive_colors (bool): If True, the colors are exclusive. If False, the colors are not exclusive.
        num_samples (int): Number of samples to use.
        input_length (int): Number of frames to use as input.for the model.
        frame_stride (int): Stride to use when reading in the frames.
        feature_dim (int): Dimension of the feature vector. If None, the feature vector is not used.
        transform (callable, optional): Optional transform to be applied on a sample.
    """
    def __init__(self, path, part, visible_objects, different_actions, different_colors, different_objects,
                 exclusive_colors, num_samples, input_length=16, frame_stride=1, feature_dim=None, transform=None):

        assert isinstance(path, str) and isinstance(part, str)
        assert part in ["training", "validation", "constant-test", "generalization-test"]
        assert isinstance(visible_objects, list) and len(visible_objects) <= 6

        if part == "training":
            max_samples_per_dir = 5000
        elif part == "validation":
            max_samples_per_dir = 2500
        elif part == "constant-test" or part == "generalization-test":
            max_samples_per_dir = 2000
        else:
            raise ValueError("Wrong part parameter. This dataset is not available!")

        self.path = path[:-1] if path[-1] == "/" else path # instead of removesuffix for python version >= 3.9
        self.part = part
        self.visible_objects = visible_objects
        self.different_actions = different_actions
        self.different_colors = different_colors
        self.different_objects = different_objects
        self.exclusive_colors = exclusive_colors
        self.input_length = input_length
        self.frame_stride = frame_stride
        self.num_sub_dirs = len(self.visible_objects)
        self.num_samples_per_dir = min(num_samples // self.num_sub_dirs, max_samples_per_dir)
        self.num_samples = len(self.visible_objects) * self.num_samples_per_dir
        self.transform = transform
        self.LABEL_LENGTH = 19
        self.DICTIONARY = ["put down", "picked up", "pushed left", "pushed right",
                           "apple", "banana", "cup", "football", "book", "pylon", "bottle", "star", "ring",
                           "red", "green", "blue", "yellow", "white", "brown"]

    def __len__(self):
        return self.num_samples

    def __getitem__(self, item):
        # label: path/Vi-Cc-Oo/part/sequence_xxxx/label.npy -> one-hot-encoded
        # imgs:  path/Vi-Cc-Oo/part/sequence_xxxx/frame_bbbbbb.png - frame_eeeeee.png
        # joints:path/Vi-Cc-Oo/part/sequence_xxxx/frame_bbbbbb.txt - frame_eeeeee.txt

        dir_number = self.visible_objects[item // self.num_samples_per_dir]
        sequence_number = item % self.num_samples_per_dir
        if self.part == "constant-test":
            dir_path = f"{self.path}/{self.part}/V{dir_number}-test"
        elif self.part == "generalization-test":
            dir_path = f"{self.path}/{self.part}/V{dir_number}-generalization-test"
        else:
            dir_path = f"{self.path}/V{dir_number}-A{self.different_actions}-C{self.different_colors}-O{self.different_objects}{'-X' if self.exclusive_colors else ''}/{self.part}"
        sequence_path = f"{dir_path}/sequence_{sequence_number:04d}"

        # reading sentence out of label.npy - NOT one-hot-encoded
        label = torch.from_numpy(np.load(f"{sequence_path}/label.npy")).to(dtype=torch.long)
        joint_paths = glob.glob(f"{sequence_path}/joints_*.npy")
        frame_paths = glob.glob(f"{sequence_path}/frame_*.png")

        # glob returns unordered
        joint_paths.sort()
        frame_paths.sort()

        num_frames = len(frame_paths)
        assert num_frames > 0

        # Pad all sequences to the same length
        # If sequence is longer take the last input_length sequences
        # If sequence is shorter repeat the first frame input_length-num_frames times
        joints = torch.zeros(self.input_length, 6, dtype=torch.float32) # 6 joints
        frames = torch.zeros(self.input_length, 3, 224, 398, dtype=torch.float32)  # img shape (3, 224, 398) TODO: remove magic numbers

        input_index = self.input_length -1
        for n in range(num_frames-1, -1, -self.frame_stride):
            joint_path = joint_paths[n]
            frame_path = frame_paths[n]

            joints[input_index] = torch.from_numpy(np.load(joint_path)).to(torch.float32)
            frames[input_index] = read_image(frame_path).to(torch.float32) / 255
            if self.transform is not None:
                frames[input_index] = self.transform(frames[input_index])
            input_index -= 1
            # if input is filled, return
            if input_index < 0:
                self.assert_output(frames, joints, label)
                return frames, joints, label

        # if sequence is shorter than input_length, repeat first frame
        frame0 = read_image(frame_paths[0]).to(torch.float32) / 255
        if self.transform is not None:
            frame0 = self.transform(frame0)
        joint0 = torch.from_numpy(np.load(joint_paths[0])).to(torch.float32)
        for i in range(input_index, -1, -1):
            frames[i] = frame0
            joints[i] = joint0

        self.assert_output(frames, joints, label) #TODO: add debug mode
        return frames, joints, label


    def get_sentence_string(self, label):
        return f"{self.DICTIONARY[int(label[0])]} {self.DICTIONARY[int(label[1])]} {self.DICTIONARY[int(label[3])]}"

    def assert_output(self, frames, joints, label):
        # frames.shape -> (frames, 3, 224, 398)
        # joints.shape -> (frames, 6)
        # label.shape  -> (3) : word tokens
        # all have dtype=torch.float32
        assert frames.shape[0] == self.input_length , f"frames.shape[0] = {frames.shape[0]} != {self.input_length}"
        assert joints.shape[0] == self.input_length, f"joints.shape[0] = {joints.shape[0]} != {self.input_length}"
        assert frames.shape[1] == 3 , f"frames.shape[1] = {frames.shape[1]} != 3"
        assert frames.shape[2] == 224, f"frames.shape[2] = {frames.shape[2]} != 224"
        assert frames.shape[3] == 398, f"frames.shape[3] = {frames.shape[3]} != 398"
        assert joints.shape[1] == 6, f"joints.shape[1] = {joints.shape[1]} != 6"
        assert label.shape[0] == 3, f"label.shape[0] = {label.shape[0]} != 3"
        assert frames.dtype == torch.float32, f"frames.dtype = {frames.dtype} != torch.float32"
        assert joints.dtype == torch.float32, f"joints.dtype = {joints.dtype} != torch.float32"
        # assert label.dtype == torch.float32, f"label.dtype = {label.dtype} != torch.float32"
        # TODO: label is int64 !? -> why?
        # I didn't change change anything, so it must be the same in the original code
        # TODO: can we use uint8 instead?


