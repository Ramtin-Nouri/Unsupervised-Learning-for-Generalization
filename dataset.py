import glob
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from torchvision.io import read_image
from pytorch_lightning import LightningDataModule
from torchvision import transforms

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
                 exclusive_colors, num_samples, input_length=16, frame_stride=1, transform=None, debug=False, offset=0):

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
        self.offset = offset

    def __len__(self):
        """Returns the number of samples in the dataset."""
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
        sequence_path = f"{dir_path}/sequence_{item+self.offset:04d}"

        # reading sentence out of label.npy - NOT one-hot-encoded
        label = torch.from_numpy(np.load(f"{sequence_path}/label.npy")).to(dtype=torch.uint8)
        joint_paths = glob.glob(f"{sequence_path}/joints_*.npy")
        frame_paths = glob.glob(f"{sequence_path}/frame_*.png")

        # glob returns unordered
        joint_paths.sort()
        frame_paths.sort()


        """Fix dataset issues.
        1. Some sequences are reset after a certain number of frames. 
        The frames before the reset need to be excluded.
        2. Some sequences have unrelated "stray" frames after the correct sequence.
        The action.info files contain the correct number of frames and the resets.
        """
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

        """Pad all sequences to the same length.
        If sequence is longer take the last input_length sequences
        I.e. if input_length=16 and num_frames=20, the last 16 frames are taken

        If sequence is shorter repeat the first frame input_length-num_frames times
        I.e. if input_length=16 and num_frames=10, the first frame is repeated 6 times

        If stride is given, the input sequences are subsampled
        """
        joints = torch.zeros(self.input_length, self.JOINTS, dtype=torch.float32) # 6 joints
        frames = torch.zeros(self.input_length, self.CHANNELS, self.HEIGHT, self.WIDTH, dtype=torch.float32)  # i.e. shape (16, 3, 224, 398)

        input_index = self.input_length -1
        for n in range(num_frames-1, -1, -self.frame_stride):
            frames[input_index], joints[input_index] = self.get_image(frame_paths, joint_paths, n)
            
            input_index -= 1
            # if input is filled, return
            if input_index < 0:
                if self.debug:
                    self.assert_output(frames, joints, label)
                return frames, joints, label

        # if sequence is shorter than input_length, repeat first frame
        frame0, joint0 = self.get_image(frame_paths, joint_paths, 0)
        for i in range(input_index, -1, -1):
            frames[i] = frame0
            joints[i] = joint0

        if self.debug:
            self.assert_output(frames, joints, label)
        return frames, joints, label

    def get_image(self, frame_paths, joint_paths, index):
        frame_path = frame_paths[index]
        frame = read_image(frame_path).to(torch.float32) / 255
        if self.transform is not None:
            frame = self.transform(frame)

        joint_path = joint_paths[index]
        joint = torch.from_numpy(np.load(joint_path)).to(torch.float32)
        return frame, joint


    def get_sentence_string(self, label):
        return f"{self.DICTIONARY[int(label[0])]} {self.DICTIONARY[int(label[1])]} {self.DICTIONARY[int(label[3])]}"

    def assert_output(self, frames, joints, label):
        # frames.shape -> (frames, 3, 224, 398)
        # joints.shape -> (frames, 6)
        # label.shape  -> (3) : word tokens
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

class DataModule(LightningDataModule):
    """ DataModule for the MultimodelSimulation dataset.
    For unsupervised data will concatenate all dataset configurations 
    as well as provide some test data from the generalization test set and constant test set.

    Args:
        config (dict): configuration dictionary
        unsupervised (bool): use unsupervised learning
    """
    def __init__(self, config, unsupervised=False):
        super().__init__()
        self.config = config
        self.unsupervised = unsupervised
        
        dataset_mean = [0.7605, 0.7042, 0.6045]
        dataset_std = [0.1832, 0.2083, 0.2902]
        normal_transform = transforms.Normalize(mean=dataset_mean, std=dataset_std)
        self.transform = normal_transform
        
        self.train_loader = None
        self.val_loaders = None

    def prepare_data(self):
        pass

    def setup(self, stage=None):
        """ Setup the train, val and test datasets."""
        if self.unsupervised:
            # For unsupervised training we use the whole dataset
            train_datasets=[]
            val_datasets=[]
            for visible_objects in [1,2,6]:
                for colors in [1,6]:
                    for exclusive_colors in [False,True]:
                        if exclusive_colors:
                            different_objects = 4
                        else:
                            different_objects = 9


                        train_datasets.append(self.create_dataset(visible_objects, colors, different_objects, exclusive_colors, "training"))
                        val_datasets.append(self.create_dataset(visible_objects, colors, different_objects, exclusive_colors, "validation"))

            training_data = ConcatDataset(train_datasets)
            validation_data = ConcatDataset(val_datasets)
            print(f"Training data consists of {len(train_datasets)} datasets with {len(training_data)} samples")
            print(f"Validation data consists of {len(val_datasets)} datasets with {len(validation_data)} samples")
            test_datasets=[]
            for V in range(1,7):
                test_datasets.append(self.create_dataset(V, 0, 0, False, "constant-test"))
                test_datasets.append(self.create_dataset(V, 0, 0, False, "generalization-test"))
            test_data = ConcatDataset(test_datasets)
            print(f"Test data consists of {len(test_datasets)} datasets with {len(test_data)} samples")
            self.test_loader = DataLoader(test_data, batch_size=1, shuffle=False, num_workers=self.config["num_workers"])
        else:
            visible_objects = self.config["visible_objects"]
            colors = self.config["different_colors"]
            different_objects = self.config["different_objects"]
            exclusive_colors = self.config["exclusive_colors"]

            training_data = self.create_dataset(visible_objects, colors, different_objects, exclusive_colors, "training")
            validation_data = self.create_dataset(visible_objects, colors, different_objects, exclusive_colors, "validation")
            generalization_validation_data = self.create_dataset(visible_objects, 0, 0, False, "generalization-validation")

        # dataloader
        self.train_loader = DataLoader(dataset=training_data, batch_size=self.config["batch_size"], shuffle=True,
                                num_workers=self.config["num_workers"])
        val_loader = DataLoader(dataset=validation_data, batch_size=self.config["batch_size"], shuffle=False,
                                num_workers=self.config["num_workers"])
        if self.unsupervised:
            generalization_val_loader = None
            self.val_loaders = val_loader
        else:
            generalization_val_loader = DataLoader(dataset=generalization_validation_data, batch_size=self.config["batch_size"],
                                shuffle=False, num_workers=self.config["num_workers"])
            self.val_loaders = [val_loader, generalization_val_loader]


    def create_dataset(self, visible_objects, different_colors, different_objects, exclusive_colors, part):
        """ Create a dataset for a specific configuration.
        
        Just a wrapper for the MultimodalSimulation class, that uses some globally defined parameters.
        """
        offset = 0
        if self.unsupervised:
            if part == "training":
                num_samples = self.config["num_training_samples_unsupervised"]
            elif part == "validation":
                num_samples = self.config["num_validation_samples_unsupervised"]
            else:
                num_samples = 2 # We only need 2 samples for the test set
        else:
            if part == "training":
                num_samples = self.config["num_training_samples"]
            elif part == "validation":
                num_samples = self.config["num_validation_samples"]
            elif part == "generalization-validation":
                num_samples = 500
                part="generalization-test"
            elif part == "generalization-test":
                num_samples = 1500
                offset = 500
            elif part == "constant-test":
                num_samples = 2000
            else:
                raise ValueError(f"Unknown part {part}")

        if self.config["debug"]:
            num_samples = min(10,num_samples)

        return MultimodalSimulation(path=self.config["data_path"],
                                    visible_objects=visible_objects,
                                    different_actions=4,
                                    different_colors=different_colors,
                                    different_objects=different_objects,
                                    exclusive_colors=exclusive_colors,
                                    part=part,
                                    num_samples=num_samples,
                                    input_length=self.config["input_length"],
                                    frame_stride=self.config["input_stride"],
                                    transform=self.transform,
                                    debug=self.config["debug"],
                                    offset=offset)

    def train_dataloader(self):
        """ Return the train dataloader."""
        return self.train_loader

    def val_dataloader(self):
        """ Return the validation dataloader."""
        return self.val_loaders
    
    def test_dataloader(self):
        """ Return the test dataloader."""
        return self.test_loader

