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
    
    Args:
        path (str): Path to the dataset.
        part (str): Part of the dataset to use. Can be "training", "validation", "constant-test" or "generalization-test".
        visible_objects (int): Number of simultaneously visible objects.
        different_actions (int): Number of different actions. (actually always 4)
        different_colors (int): Number of different colors each object can have.
        different_objects (int): Number of different object types.
        exclusive_colors (bool): If True, the colors are exclusive to the objects.
        num_samples (int): Number of samples to use.
        input_length (int): Number of frames to use as input.for the model.
        frame_stride (int): Stride to use when reading in the frames.
        transform (callable, optional): Optional transform to be applied on a sample.

    Attributes:
        DICTIONARY (list): List of words in the dictionary.
        CHANNELS (int): Number of channels in the images.
        HEIGHT (int): Height of the images.
        WIDTH (int): Width of the images.
        JOINTS (int): Number of joints in the robot arm.
    """

    WIDTH = 398
    HEIGHT = 224
    CHANNELS = 3
    JOINTS = 6
    DICTIONARY = ["put down", "picked up", "pushed left", "pushed right",
                           "apple", "banana", "cup", "football", "book", "pylon", "bottle", "star", "ring",
                           "red", "green", "blue", "yellow", "white", "brown"]
    LABEL_LENGTH = 19
    def __init__(self, path, part, visible_objects, different_actions, different_colors, different_objects,
                 exclusive_colors, num_samples, input_length=16, frame_stride=1, transform=None, debug=False):

        assert isinstance(path, str) and isinstance(part, str)
        assert part in ["training", "validation", "constant-test", "generalization-test"]
        assert isinstance(visible_objects, int) and visible_objects <= 6

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
        self.num_samples = min(num_samples, max_samples_per_dir)
        self.transform = transform
        self.debug = debug

    def __len__(self):
        """Returns the number of samples in the dataset."""
        if self.debug:
            return 20
        return self.num_samples

    def __getitem__(self, item):
        """Returns a sample from the dataset.
        One sample consists of the frames, the joints and the label.
        The frames are padded with the first frame to the input_length.
        If a stride is given, the input sequences are subsampled.
        
        The data is read from the disk on the fly.
        label: path/Vi-Cc-Oo/part/sequence_xxxx/label.npy -> one-hot-encoded
        imgs:  path/Vi-Cc-Oo/part/sequence_xxxx/frame_bbbbbb.png - frame_eeeeee.png
        joints:path/Vi-Cc-Oo/part/sequence_xxxx/frame_bbbbbb.txt - frame_eeeeee.txt

        Args:
            item (int): Index of the sample to return.
        """

        dir_number = self.visible_objects
        if self.part == "constant-test":
            dir_path = f"{self.path}/{self.part}/V{dir_number}-test"
        elif self.part == "generalization-test":
            dir_path = f"{self.path}/{self.part}/V{dir_number}-generalization-test"
        else:
            dir_path = f"{self.path}/V{dir_number}-A{self.different_actions}-C{self.different_colors}-O{self.different_objects}{'-X' if self.exclusive_colors else ''}/{self.part}"
        sequence_path = f"{dir_path}/sequence_{item:04d}"

        # reading sentence out of label.npy - NOT one-hot-encoded
        label = torch.from_numpy(np.load(f"{sequence_path}/label.npy")).to(dtype=torch.uint8)
        joint_paths = glob.glob(f"{sequence_path}/joints_*.npy")
        frame_paths = glob.glob(f"{sequence_path}/frame_*.png")

        # glob returns unordered
        joint_paths.sort()
        frame_paths.sort()


        # Fix dataset issues
        # 1. Some sequences are reset after a certain number of frames. 
        # The frames before the reset need to be excluded.
        # 2. Some sequences have unrelated "stray" frames after the correct sequence.
        # The action.info files contain the correct number of frames and the resets.
        with open(f"{sequence_path}/action.info", "r") as f:
            action_info = f.read().splitlines()
            correct_number_frames =int(action_info[0].split("[")[1].split(",")[0])
            
            if len(action_info) > 2 and "Resetting" in action_info[2]:
                last_reset = int(action_info[2].split("[")[1].split("]")[0].split(",")[-1])
                frame_paths = frame_paths[last_reset:correct_number_frames]
            else:
                frame_paths = frame_paths[:correct_number_frames]




        num_frames = len(frame_paths)
        assert num_frames > 0

        # Pad all sequences to the same length
        # If sequence is longer take the last input_length sequences
        # If sequence is shorter repeat the first frame input_length-num_frames times
        joints = torch.zeros(self.input_length, self.JOINTS, dtype=torch.float32) # 6 joints
        frames = torch.zeros(self.input_length, self.CHANNELS, self.HEIGHT, self.WIDTH, dtype=torch.float32)  # img shape (3, 224, 398)

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

        if self.debug:
            self.assert_output(frames, joints, label)
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
        assert frames.shape[1] == self.CHANNELS , f"frames.shape[1] = {frames.shape[1]} != 3"
        assert frames.shape[2] == self.HEIGHT, f"frames.shape[2] = {frames.shape[2]} != 224"
        assert frames.shape[3] == self.WIDTH, f"frames.shape[3] = {frames.shape[3]} != 398"
        assert joints.shape[1] == self.JOINTS, f"joints.shape[1] = {joints.shape[1]} != 6"
        assert label.shape[0] == 3, f"label.shape[0] = {label.shape[0]} != 3"
        assert frames.dtype == torch.float32, f"frames.dtype = {frames.dtype} != torch.float32"
        assert joints.dtype == torch.float32, f"joints.dtype = {joints.dtype} != torch.float32"
        assert label.dtype == torch.uint8, f"label.dtype = {label.dtype} != torch.uint8"