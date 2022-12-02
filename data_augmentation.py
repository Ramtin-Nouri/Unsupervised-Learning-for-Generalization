import torch
from torch import Tensor
import torch.nn as nn
from torchvision import transforms as T
from torchvision.transforms import functional_tensor as F
from torchvision.utils import _log_api_usage_once

class DataAugmentation():
    """ Data augmentation class for the training set.
    
    Provides a method to apply a set of data augmentation transformations to a
    given image.
    """

    def __init__(self):
        _log_api_usage_once(self)
        

    def __call__(self, batch, augment_action, augment_color, augment_object):
        """ Applies the data augmentation transformations to a batch of images.

        Args:
            batch (torch.Tensor): Batch of image sequences to apply the transformations to. Shape: (B, T, C, H, W)

        Returns:
            torch.Tensor: Batch of transformed image sequences. Shape: (B, T, C, H, W)
        """
        # apply transformations
        compose = self.create_compose(augment_action, augment_color, augment_object)
        for i in range(batch.size(0)):
            batch[i] = compose(batch[i])
        return batch

    def create_compose(self, augment_action, augment_color, augment_object):
        """ Creates a composition of data augmentation transformations.

        Args:
            augment_action (bool): Whether to apply the action augmentation.
            augment_color (bool): Whether to apply the color augmentation.
            augment_object (bool): Whether to apply the object augmentation.

        Returns:
            torchvision.transforms.Compose: Composition of the transformations to apply.
        """
        transformations = []
        # add generic transformations
        transformations.append(T.RandomRotation(degrees=(0, 25)))
        if augment_action:
            # if masking action, randomly flip
            transformations.append(T.RandomHorizontalFlip(p=0.3))
            # transformations.append(RandomSequenceFlip(p=0.3))
        
        if augment_color:
            # if masking color, randomly change color
            transformations.append(T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.5))
            transformations.append(T.RandomGrayscale(p=0.1))

        if augment_object:
            # if masking object, randomly blur
            transformations.append(T.GaussianBlur(kernel_size=7, sigma=(3)))

        return T.Compose(transformations)



class RandomSequenceFlip(nn.Module):
    """Temporally flip the given image sequence randomly with a given probability.
    If the image is torch Tensor, it is expected
    to have [...,T, C, H, W] shape, where ... means an arbitrary number of leading
    dimensions.
    Implementation very similar to the one in torchvision.transforms.RandomHorizontalFlip.

    Args:
        index (int): The dimension to reverse.
        p (float): probability of the image being flipped. Default value is 0.5
    """
    def __init__(self,  p=0.5):
        super().__init__()
        _log_api_usage_once(self)
        self.p = p
        
    def forward(self, img):
        if torch.rand(1) < self.p:
            return self.tflip(img)
        return img

    def __repr__(self):
        return f'{self.__class__.__name__}(p={self.p})'

    def tflip(self, img: Tensor) -> Tensor:
        """Flip the given image sequence temporally.
        If the image is torch Tensor, it is expected
        to have [...,T, C, H, W] shape, where ... means an arbitrary number of leading
        dimensions.
        Implementation very similar to the one in torchvision.transforms.functional_tensor.hflip.

        Args:
            img (Tensor): Image to be flipped.

        Returns:
            Tensor:  Temporally flipped image.
        """
        F._assert_image_tensor(img)

        return img.flip(-4)