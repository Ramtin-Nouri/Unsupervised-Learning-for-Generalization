import torch
from torch import Tensor
import torch.nn as nn
from torchvision import transforms as T
from torchvision.transforms import functional as F
from torchvision.utils import _log_api_usage_once
from typing import List, Tuple

class DataAugmentation():
    """ Data augmentation class for the training set.
    
    Provides a method to apply a set of data augmentation transformations to a
    given image.
    """

    def __init__(self):
        _log_api_usage_once(self)
        

    def __call__(self, batch, augment_action, augment_color, augment_object, augment_material):
        """ Applies the data augmentation transformations to a batch of images.

        Args:
            batch (torch.Tensor): Batch of image sequences to apply the transformations to. Shape: (B, T, C, H, W)

        Returns:
            torch.Tensor: Batch of transformed image sequences. Shape: (B, T, C, H, W)
        """
        # apply transformations
        # TODO: add material augmentation
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
        
        transformations.append(T.RandomRotation(degrees=(0, 25)))
        if augment_action:
            # if masking action, randomly flip
            transformations.append(T.RandomHorizontalFlip(p=0.3))
        
        if augment_color:
            # if masking color, randomly change color
            # transformations.append(RandomColorShift(0.2))
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
        Implementation very similar to the one in torchvision.transforms.functional.hflip.

        Args:
            img (Tensor): Image to be flipped.

        Returns:
            Tensor:  Temporally flipped image.
        """
        return img.flip(-4)

class Affine(nn.Module):
    """Variation of PyTorch's affine transformation that keeps relevant pixels in the image.

    The maximum scale and translation are calculated based on the sampled angle.
    Some of the values are hard-coded for simplicity and may not generalize well to other image sizes.
    
    Args:
        degrees (list): Range of degrees to sample from.
    """
    def __init__(self,sigma):
        super().__init__()
        self.sigma = sigma

    @staticmethod
    def get_params(sigma: float) -> Tuple[float, float, Tuple[float, float]]:
        """Get parameters for affine transformation.

        Returns:
            float: angle parameter to be passed to rotate
            float: scale parameter to be passed to rotate
            tuple: translate parameter to be passed to rotate
        """
        angle = float(torch.empty(1).normal_(0, sigma).item())
        if angle < 0:
            angle += 360 # angle must be positive for calculations below
            
        min_scale = 0.5
        if angle < 90:
            max_scale = 1.2 - min_scale* (angle)/90
        elif angle < 180:
            max_scale = 1.2 - min_scale* (180-angle)/90
        elif angle < 270:
            max_scale = 1.2 - min_scale* (270-angle)/90
        else:
            max_scale = 1.2 - min_scale* (360-angle)/90

        scale = torch.empty(1).uniform_(min_scale, max_scale).item()
        
        max_dx = 150*(1.2-scale)
        max_dy = 10
        tx = int(round(torch.empty(1).uniform_(-max_dx, max_dx).item()))
        ty = int(round(torch.empty(1).uniform_(-max_dy, max_dy).item()))
        translations = (tx, ty)
        return angle, scale, translations
        
    def forward(self, x):
        angle, scale, translate = self.get_params(self.sigma)
        return F.affine(x, angle=angle, translate=translate, scale=scale, shear=0)

class RandomColorShift(nn.Module):
    """Randomly shift the color channels of the given image sequence.

    Args:
        sigma (float): Standard deviation of the normal distribution to sample from.
    """

    def __init__(self, sigma):
        super().__init__()
        self.sigma = sigma
        
    def forward(self, img):
        for c in range(3):
            r = float(torch.empty(1).normal_(1, self.sigma).item())
            r = max(0,r) # cannot be <0
            img[c] = F.adjust_brightness(img[c],r)
        return img

    def __repr__(self):
        return f'{self.__class__.__name__}(sigma={self.sigma})'
