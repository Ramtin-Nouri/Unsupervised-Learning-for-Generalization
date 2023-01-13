import torch
from torch import nn
import torchvision
from pytorch_lightning import LightningModule
from helper import *
import numpy as np
from data_augmentation import DataAugmentation

class SwinTransformer(LightningModule):
    """ Swin Transformer model for classification.

    Args:
        config (dict): Dictionary containing the configuration parameters.
    """
    def __init__(self, config):
        super().__init__()
        self.save_hyperparameters()

        self.label_size = config["dictionary_size"]
        self.use_joints = config["use_joints"]
        self.learning_rate = config["learning_rate"]
        self.num_joints = config["num_joints"]
        self.sentence_length = config["sentence_length"]
        self.width = config["width"]
        self.height = config["height"]
        self.use_augmentation = config["data_augmentation"]
        if self.use_augmentation:
            self.augmentation = DataAugmentation()

        self.masks = [[0,0,1], [0,1,0], [1,0,0], [0,1,1], [1,0,1], [1,1,0], [1,1,1]]

        self.conv3d = nn.Conv3d(6,3,2) # RGB+Mask -> 3 channels
        self.transformer = torchvision.models.video.swin3d_b()
        self.hidden_size = self.transformer.head.out_features
        self.linear1 = nn.Linear(self.hidden_size, config["dictionary_size"])
        self.linear2 = nn.Linear(self.hidden_size, config["dictionary_size"])
        self.linear3 = nn.Linear(self.hidden_size, config["dictionary_size"])

        self.loss_fn = nn.CrossEntropyLoss()
        self.reset_metrics_train()
        self.reset_metrics_val()
        
    def forward(self, x_frames, mask,joints=None):
        """ Forward pass of the model.

        Args:
            x_frames (torch.Tensor): Tensor containing the frames of the video. Shape: (batch_size, num_frames, 3, height, width) i.e. (batch_size, num_frames, 3, 224, 398)

        Returns:
            torch.Tensor: Output of the model. Shape: (batch_size, sentence_length, label_size) i.e. (batch_size, 3, 19)
        """
        # encode frames
        if self.use_joints:
           raise NotImplementedError("Not implemented yet")

        # find size of different input dimensions
        b, seq_len, c, h, w = x_frames.size()
        # expand mask to match the size of the input
        mask_expanded = mask.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, -1, h, w)
        # concatenate mask with the input
        x = torch.cat((x_frames, mask_expanded), dim=2)
        x = x.permute(0, 2, 1, 3, 4)

        x = self.conv3d(x) # (batch_size, num_frames, 6, height, width) -> (batch_size, num_frames, 3, height, width)
        x = self.transformer(x) # (batch_size, hidden_size) i.e. (batch_size, 400)

        pred1 = self.linear1(x) # (batch_size, label_size) i.e. (batch_size, 19)
        pred2 = self.linear2(x) # (batch_size, label_size) i.e. (batch_size, 19)
        pred3 = self.linear3(x) # (batch_size, label_size) i.e. (batch_size, 19)

        stack = torch.stack([pred1, pred2, pred3], dim=1) # (batch_size, sentence_length, label_size) i.e. (batch_size, 3, 19)
        return stack
        

    def configure_optimizers(self):
        """ Configure the optimizer.

        Returns:
            torch.optim.Optimizer: Optimizer.
        """
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer
    
    def loss(self, output, labels, mask):
        """ Calculate the loss.

        Args:
            output (torch.Tensor): Output of the model. Shape: (batch_size, sentence_length, label_size) i.e. (batch_size, 3, 19)
            labels (torch.Tensor): Labels of the data. Shape: (batch_size, sentence_length) i.e. (batch_size, 3)
            mask (torch.Tensor): Mask of the data. Shape: (batch_size, sequence_length, sentence_length) i.e. (batch_size, 16, 3)

        Returns:
            torch.Tensor: Loss.
        """
        loss = torch.zeros(1, device=output.device)
        for i in range(self.sentence_length):
            l = self.loss_fn(output[:, i, :], labels[:, i])
            loss += l * mask[:, 0, i]
        
        return loss

    def training_step(self, batch, batch_idx):
        """ Training step.

        Args:
            batch (tuple): Tuple containing the frames, joints and the labels.
                frames (torch.Tensor): Tensor containing the frames of the video. Shape: (batch_size, num_frames, 3, height, width) i.e. (batch_size, num_frames, 3, 224, 398)
                joints (torch.Tensor): Tensor containing the joints of the video. Shape: (batch_size, num_frames, num_joints) i.e. (batch_size, num_frames, 6)
                labels (torch.Tensor): Labels of the data. Shape: (batch_size, sentence_length) i.e. (batch_size, 3)
            batch_idx (int): Index of the batch.

        Returns:
            torch.Tensor: Loss.
        """
        frames, joints, labels = batch
        # TODO: add joints
        
        # random mask for each input
        mask = [ [self.masks[np.random.choice(len(self.masks))]] * frames.size(1) ]*frames.size(0) #each batch and image has the same mask
        mask = torch.tensor(mask, device=frames.device)

        if self.use_augmentation:
            augment_action = not mask[0][0][0]
            augment_color = not mask[0][0][1]
            augment_object = not mask[0][0][2]
            frames = self.augmentation(frames, augment_action, augment_color, augment_object)

        
        output = self(frames, mask)
        loss = self.loss(output, labels, mask)
        self.log('train_loss', loss)
        self.calculate_accuracy(output, labels, train=True)
        return loss

    def training_epoch_end(self, outputs):
        """ Log accuracies at the end of the epoch.

        Args:
            outputs (list): List containing the outputs of the training steps.
        """
        epoch_action_acc = self.training_action_correct * 100 / self.training_total
        epoch_color_acc = self.training_color_correct * 100 / self.training_total
        epoch_object_acc = self.training_object_correct * 100 / self.training_total
        epoch_acc = (epoch_action_acc + epoch_color_acc + epoch_object_acc) / 3
        self.log('train_acc_action', epoch_action_acc, on_step=False, on_epoch=True)
        self.log('train_acc_color', epoch_color_acc, on_step=False, on_epoch=True)
        self.log('train_acc_object', epoch_object_acc, on_step=False, on_epoch=True)
        self.log('train_acc', epoch_acc, on_step=False, on_epoch=True)
        self.reset_metrics_train()
        print_with_time(f"Epoch {self.current_epoch} train acc: {epoch_acc}")
    
    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        """ Same as training step."""
        frames, joints, labels = batch
        # TODO: add joints
        mask = torch.ones(frames.size(0), frames.size(1), 3, device=frames.device)

        output = self(frames, mask)
        loss = self.loss(output, labels, mask)
        lossname = f'val_loss'
        if dataloader_idx == 1:
            lossname += '_gen'
        self.log(lossname, loss)
        
        if dataloader_idx == 0:
            self.calculate_accuracy(output, labels, train=False, gen=False)
        else:
            self.calculate_accuracy(output, labels, train=False, gen=True)
        return loss
    
    def validation_epoch_end(self, outputs):
        """ Same as training epoch end."""
        epoch_action_acc = self.val_action_correct * 100 / self.val_total
        epoch_color_acc = self.val_color_correct * 100 / self.val_total
        epoch_object_acc = self.val_object_correct * 100 / self.val_total
        epoch_acc = (epoch_action_acc + epoch_color_acc + epoch_object_acc) / 3
        self.log('val_acc_action', epoch_action_acc, on_step=False, on_epoch=True)
        self.log('val_acc_color', epoch_color_acc, on_step=False, on_epoch=True)
        self.log('val_acc_object', epoch_object_acc, on_step=False, on_epoch=True)
        self.log('val_acc', epoch_acc, on_step=False, on_epoch=True)

        # generalization validation
        epoch_action_acc = self.val_gen_action_correct * 100 / self.val_gen_total
        epoch_color_acc = self.val_gen_color_correct * 100 / self.val_gen_total
        epoch_object_acc = self.val_gen_object_correct * 100 / self.val_gen_total
        epoch_acc = (epoch_action_acc + epoch_color_acc + epoch_object_acc) / 3
        self.log('val_acc_action_gen', epoch_action_acc, on_step=False, on_epoch=True)
        self.log('val_acc_color_gen', epoch_color_acc, on_step=False, on_epoch=True)
        self.log('val_acc_object_gen', epoch_object_acc, on_step=False, on_epoch=True)
        self.log('val_acc_gen', epoch_acc, on_step=False, on_epoch=True)

        self.reset_metrics_val()
        print_with_time(f"Epoch {self.current_epoch} val_acc: {epoch_acc}")

    def calculate_accuracy(self, output, labels, train=True, gen=False):
        """ Calculate the accuracy of the model.

        Args:
            output (torch.Tensor): Output of the model. Shape: (batch_size, sentence_length, label_size) i.e. (batch_size, 3, label_size)
            labels (torch.Tensor): Labels of the data. Shape: (batch_size, sentence_length)
            train (bool): Whether to calculate the training accuracy or the validation accuracy.
        """
        _, action_output_batch = torch.max(output[:, 0, :], dim=1)
        _, color_output_batch = torch.max(output[:, 1, :], dim=1)
        _, object_output_batch = torch.max(output[:, 2, :], dim=1)

        if train:
            self.training_action_correct += torch.sum(action_output_batch == labels[:, 0])
            self.training_color_correct += torch.sum(color_output_batch == labels[:, 1])
            self.training_object_correct += torch.sum(object_output_batch == labels[:, 2])

            self.training_total += labels.shape[0]
        elif gen:
            # generalization validation
            self.val_gen_action_correct += torch.sum(action_output_batch == labels[:, 0])
            self.val_gen_color_correct += torch.sum(color_output_batch == labels[:, 1])
            self.val_gen_object_correct += torch.sum(object_output_batch == labels[:, 2])

            self.val_gen_total += labels.shape[0]
        else:
            # normal validation
            self.val_action_correct += torch.sum(action_output_batch == labels[:, 0])
            self.val_color_correct += torch.sum(color_output_batch == labels[:, 1])
            self.val_object_correct += torch.sum(object_output_batch == labels[:, 2])

            self.val_total += labels.shape[0]


    def reset_metrics_train(self):
        """Reset metrics for training"""
        self.training_action_correct = 0
        self.training_color_correct = 0
        self.training_object_correct = 0
        self.training_total = 0
    
    def reset_metrics_val(self):
        """Reset metrics for validation"""
        # normal validation
        self.val_action_correct = 0
        self.val_color_correct = 0
        self.val_object_correct = 0
        self.val_total = 0
        # generalization validation
        self.val_gen_action_correct = 0
        self.val_gen_color_correct = 0
        self.val_gen_object_correct = 0
        self.val_gen_total = 0


def get_layers_until(model, layer_name):
    """ Get all layers until a certain layer.
    inspired by https://github.com/knowledgetechnologyuhh/grid-3d/blob/65856bd8bd68192807e477b8ad250dc12699ef2d/grid3d/utils.py#L30

    Args:
        model (torch.nn.Module): Model.
        layer_name (str): Name of the layer.

    Returns:
        list: List of layers until the layer with the given name.
    """
    layers = list(model._modules.keys())
    layer_count = 0
    for layer in layers:
        if layer != layer_name:
            layer_count += 1
        else:
            break
    for i in range(1, len(layers) - layer_count):
        model._modules.pop(layers[-i])
    feature_extractor = nn.Sequential(model._modules)
    for param in feature_extractor.parameters():
        param.requires_grad = False
    feature_extractor.eval()
    return feature_extractor
