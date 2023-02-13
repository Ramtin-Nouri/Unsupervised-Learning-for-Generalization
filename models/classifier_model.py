import torch
from torch import nn
import torchmetrics
from einops import rearrange
from collections import OrderedDict
from models.lstm_autoencoder import LstmEncoder
from pytorch_lightning import LightningModule
from helper import *
from models.conv_lstm_cell import *
from torchvision.models import resnet18
import numpy as np
from data_augmentation import DataAugmentation


class ClassificationLstmDecoder(LightningModule):
    """ Decoder of the LSTM model for classification. 
    The decoder is a LSTM with a linear layer at the end.
    Its input is the hidden state of the encoder and the output of the encoder.
    The output is a sequence of labels.

    Args:
        output_size (int): Number of classes to predict.
        config (dict): Dictionary containing the configuration of the model.
    """
    def __init__(self, config):
        super().__init__()
        self.hidden_size = config["lstm_hidden_size"]
        self.num_layers=config["lstm_num_layers"]
        self.sentence_length = config["sentence_length"]
        self.label_size = config["dictionary_size"]
        height = config["height"]
        width = config["width"]
        self.use_resnet = config["use_resnet"]
        if self.use_resnet:
            height = int(round(height/16))
            width = int(round(width/16))

        num_convlstm_layers = len(config["convlstm_layers"])
        input_size = (width//2**num_convlstm_layers) * (height//2**num_convlstm_layers) * config["convlstm_layers"][-1]

        self.flatten = nn.Flatten()
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=self.hidden_size,
                            num_layers=self.num_layers, batch_first=True)
        self.linear = nn.Linear(self.hidden_size, config["dictionary_size"])

        self.dropout = None
        if config["dropout_classifier"] > 0:
            self.dropout = nn.Dropout(p=config["dropout_classifier"])

    def forward(self, x):
        """ Forward pass of the decoder.

        Args:
            hidden (tuple): Tuple containing the hidden state (hx) and the cell state (cx) of the LSTM.
                hx (torch.Tensor): Hidden state of the LSTM. Shape: (num_layers, batch_size, hidden_size)
                cx (torch.Tensor): Cell state of the LSTM. Shape: (num_layers, batch_size, hidden_size)
            x (torch.Tensor): Output of the encoder. Shape: (batch_size, features, h, w) e.g. (8, 64, 224, 398)

        Returns:
            torch.Tensor: Output of the decoder. Shape: (batch_size, sentence_length, label_size) i.e. (batch_size, 3, 19)
        """
        hidden = self.init_hidden(x.shape[0], x.device)
        x = self.flatten(x)
        if self.dropout is not None:
            x = self.dropout(x)
        x = x.unsqueeze(1)
        pred = torch.zeros((x.shape[0], self.sentence_length, self.label_size), device=x.device) 
        for i in range(self.sentence_length):
            lstm_out, hidden = self.lstm(x, hidden)
            linear_out = self.linear(lstm_out)
            pred[:, i, :] = linear_out.squeeze(1)
            
        return pred
        
    def init_hidden(self, batch_size, device):
        """ Initialize the hidden state of the LSTM.

        Args:
            batch_size (int): Batch size.
            device (torch.device): Device on which the tensors should be created.

        Returns:
            tuple: Tuple containing the hidden state (hx) and the cell state (cx) of the LSTM.
                hx (torch.Tensor): Hidden state of the LSTM. Shape: (num_layers, batch_size, hidden_size)
                cx (torch.Tensor): Cell state of the LSTM. Shape: (num_layers, batch_size, hidden_size)
        """
        return  (torch.zeros((self.num_layers, batch_size, self.hidden_size), device=device),
                torch.zeros((self.num_layers, batch_size, self.hidden_size), device=device))

class ClassificationDenseDecoder(LightningModule):
    """Dense Layer Decoder of the model for classification. 

    Args:
        output_size (int): Number of classes to predict.
        config (dict): Dictionary containing the configuration of the model.
    """
    def __init__(self, config):
        super().__init__()
        self.hidden_size=config["lstm_hidden_size"] #TODO: rename 
        self.label_size = config["dictionary_size"]

        height = config["height"]
        width = config["width"]
        self.use_resnet = config["use_resnet"]
        if self.use_resnet:
            height = int(round(height/16))
            width = int(round(width/16))

        num_convlstm_layers = len(config["convlstm_layers"])
        input_size = (width//2**num_convlstm_layers) * (height//2**num_convlstm_layers) * config["convlstm_layers"][-1]

        self.flatten = nn.Flatten()
        self.linear1 = nn.Linear(input_size, self.hidden_size)
        self.linear2 = nn.Linear(self.hidden_size, self.label_size)


    def forward(self, x):
        """ Forward pass of the decoder.

        Args:
            x (torch.Tensor): Output of the encoder. Shape: (batch_size, features, h, w) e.g. (8, 64, 224, 398)

        Returns:
            torch.Tensor: Output of the decoder. Shape: (batch_size, label_size) i.e. (batch_size, 301)
        """
        x = self.flatten(x)
        x = self.linear1(x)
        x = self.linear2(x)
        return x

class DenseClassifier(LightningModule):
    """ Dense model for classification.

    Args:
        config (dict): Dictionary containing the configuration parameters.
        encoder (DenseEncoder): Trained encoder part of the DenseAutoencoder model.
    """
    def __init__(self, config, encoder):
        super().__init__()
        self.save_hyperparameters()

        self.encoder = encoder
        self.decoder = ClassificationDenseDecoder(config)

        # TODO: add masks
        # TODO: add augmentation
        self.learning_rate = config["learning_rate"]
        self.loss_fn = nn.CrossEntropyLoss()
        self.train_mAP = torchmetrics.AveragePrecision(task="binary", num_classes=config["dictionary_size"], average="macro")
        self.val_mAP = torchmetrics.AveragePrecision(task="binary", num_classes=config["dictionary_size"], average="macro")

    def forward(self, x):
        """ Forward pass of the model.

        Args:
            x (torch.Tensor): Input image. Shape: (batch_size, channels, height, width) e.g. (8, 3, 224, 398)

        Returns:
            torch.Tensor: Output of the model. Shape: (batch_size, label_size) i.e. (batch_size, 301)
        """
        mask = torch.ones(x.size(0), 3, device=x.device) #TODO: add masks
        h_t, c_t = self.encoder.init_hidden(x.size()[0])

        # autoencoder forward
        encoder_out = self.encoder(x, mask, h_t, c_t)

        encoder_out = encoder_out[-1] # take last output of encoder
        # decode
        decoder_out = self.decoder(x=encoder_out)
        return decoder_out

    def training_step(self, batch, batch_idx):
        """ Training step of the model.

        Args:
            batch (tuple): Tuple containing the input and the target.
            batch_idx (int): Index of the batch.

        Returns:
            dict: Dictionary containing the loss.
        """
        x, y = batch
        y_hat = self(x)

        loss = self.loss_fn(y_hat, y)
        self.log("train_loss", loss)

        self.train_mAP(y_hat, y)
        self.log("train_mAP", self.train_mAP, on_step=False, on_epoch=True)

        return {"loss": loss}

    def validation_step(self, batch, batch_idx):
        """ Validation step of the model.

        Args:
            batch (tuple): Tuple containing the input and the target.
            batch_idx (int): Index of the batch.

        Returns:
            dict: Dictionary containing the loss.
        """
        x, y = batch
        y_hat = self(x)

        loss = self.loss_fn(y_hat, y)
        self.log("val_loss", loss)

        self.val_mAP(y_hat, y)
        self.log("val_mAP", self.val_mAP, on_step=False, on_epoch=True)

        return {"loss": loss}

    def configure_optimizers(self):
        """ Configures the optimizer.

        Returns:
            torch.optim.Adam: Adam optimizer.
        """
        # TODO: add learning rate scheduler
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)


class LstmClassifier(LightningModule):
    """ LSTM model for classification. 

    Args:
        config (dict): Dictionary containing the configuration parameters.
        encoder (LstmEncoder): Trained encoder part of the LstmAutoencoder model.
    """
    def __init__(self, config, encoder):
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

        self.encoder = encoder
        # TODO: if pretrained, freeze the encoder
        #self.encoder.requires_frad = False # Freeze the encoder
        self.decoder = ClassificationLstmDecoder(config)
        self.masks = [[0,0,1], [0,1,0], [1,0,0], [0,1,1], [1,0,1], [1,1,0], [1,1,1]]

        self.loss_fn = nn.CrossEntropyLoss()
        self.reset_metrics_train()
        self.reset_metrics_val()
        
    def forward(self, x_frames, mask, x_joints=None):
        """ Forward pass of the model.

        Args:
            x_frames (torch.Tensor): Tensor containing the frames of the video. Shape: (batch_size, num_frames, 3, height, width) i.e. (batch_size, num_frames, 3, 224, 398)
            x_joints (torch.Tensor): Tensor containing the joints of the video. Shape: (batch_size, num_frames, num_joints) i.e. (batch_size, num_frames, 6)

        Returns:
            torch.Tensor: Output of the model. Shape: (batch_size, sentence_length, label_size) i.e. (batch_size, 3, 19)
        """
        # encode frames
        if self.use_joints:
           raise NotImplementedError("Not implemented yet")

        # find size of different input dimensions
        b, seq_len, _, h, w = x_frames.size()

        h_t, c_t = self.encoder.init_hidden(b)

        # autoencoder forward
        encoder_out = self.encoder(x_frames, mask, h_t, c_t)

        encoder_out = encoder_out[-1]
        # decode
        decoder_out = self.decoder(x=encoder_out)
        return decoder_out

    def configure_optimizers(self):
        """ Configure the optimizer.

        Returns:
            torch.optim.Optimizer: Optimizer.
        """
        # lr scheduler
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=2, verbose=True)
        return {
            'optimizer': optimizer,
            'lr_scheduler': scheduler,
            'monitor': 'val_loss/dataloader_idx_0'
        }
    
    def loss(self, output, labels, mask):
        """ Calculate the loss.

        Args:
            output (torch.Tensor): Output of the model. Shape: (batch_size, sentence_length, label_size) i.e. (batch_size, 3, 19)
            labels (torch.Tensor): Labels of the data. Shape: (batch_size, sentence_length) i.e. (batch_size, 3)

        Returns:
            torch.Tensor: Loss.
        """
        batch_size = output.shape[0]
        loss = torch.zeros(batch_size, device=output.device)
        for i in range(self.sentence_length):
            loss += self.loss_fn(output[:, i, :], labels[:, i]) * mask[:, i]
        
        return loss.sum() / batch_size

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
        
        # random mask for each input
        mask = [self.masks[np.random.choice(len(self.masks))]] * frames.size(0) #each batch has the same mask
        mask = torch.tensor(mask, device=frames.device)

        if self.use_augmentation:
            augment_action = not mask[0][0]
            augment_color = not mask[0][1]
            augment_object = not mask[0][2]
            frames = self.augmentation(frames, augment_action, augment_color, augment_object)

        if self.use_joints:
            output = self(frames, mask, joints)
        else:
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

        mask = torch.ones(frames.size(0), 3, device=frames.device)

        if self.use_joints:
            output = self(frames, mask, joints)
        else:
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
