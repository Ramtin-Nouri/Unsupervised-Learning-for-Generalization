import torch
from torch import nn
from einops import rearrange
from collections import OrderedDict
from models.lstm_autoencoder import LstmEncoder
from pytorch_lightning import LightningModule
from helper import *


class ClassificationLstmDecoder(LightningModule):
    """ Decoder of the LSTM model for classification. 
    The decoder is a LSTM with a linear layer at the end.
    Its input is the hidden state of the encoder and the output of the encoder.
    The output is a sequence of labels.

    Args:
        output_size (int): Number of classes to predict.
        config (dict): Dictionary containing the configuration of the model.
    """
    def __init__(self, config, output_size):
        super().__init__()
        self.hidden_size = config["lstm_hidden_size"]
        self.num_layers=config["lstm_num_layers"]
        self.sentence_length = config["sentence_length"]
        self.label_size = config["dictionary_size"]

        self.lstm = nn.LSTM(input_size=self.hidden_size, hidden_size=self.hidden_size,
                            num_layers=self.num_layers, batch_first=True)
        self.linear = nn.Linear(self.hidden_size, output_size)

    def forward(self, hidden, x):
        """ Forward pass of the decoder.

        Args:
            hidden (tuple): Tuple containing the hidden state (hx) and the cell state (cx) of the LSTM.
                hx (torch.Tensor): Hidden state of the LSTM. Shape: (num_layers, batch_size, hidden_size)
                cx (torch.Tensor): Cell state of the LSTM. Shape: (num_layers, batch_size, hidden_size)
            x (torch.Tensor): Output of the encoder. Shape: (batch_size, 1, hidden_size) e.g. (32, 1, 256)

        Returns:
            torch.Tensor: Output of the decoder. Shape: (batch_size, sentence_length, label_size) i.e. (batch_size, 3, 19)
        """
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

        self.encoder = encoder
        self.encoder.requires_frad = False
        self.decoder = ClassificationLstmDecoder(output_size=self.label_size, config=config)

        self.loss_fn = nn.CrossEntropyLoss()
        self.reset_metrics_train()
        self.reset_metrics_val()
        
    def forward(self, x_frames, x_joints=None):
        """ Forward pass of the model.

        Args:
            x_frames (torch.Tensor): Tensor containing the frames of the video. Shape: (batch_size, num_frames, 3, height, width) i.e. (batch_size, num_frames, 3, 224, 398)
            x_joints (torch.Tensor): Tensor containing the joints of the video. Shape: (batch_size, num_frames, num_joints) i.e. (batch_size, num_frames, 6)

        Returns:
            torch.Tensor: Output of the model. Shape: (batch_size, sentence_length, label_size) i.e. (batch_size, 3, 19)
        """
        N = x_frames.shape[0]  # batch size
        L = x_frames.shape[1]  # num_frames

        # encode frames
        x_frames = rearrange(x_frames, 'N L c w h -> L N c w h')
        hidden = self.encoder.init_hidden(N, x_frames.device)

        if self.use_joints:
            if x_joints is None:
                print_warning("Using joints but no joints given")
                x_joints = torch.zeros((L, N, self.num_joints), device=x_frames.device)
            else:
                x_joints = rearrange(x_joints, 'n l c -> l n c', c=self.num_joints)

            for i in range(L):
                encoder_out, hidden = self.encoder(hidden, x_frames[i], x_joints[i])
        else:
            for i in range(L):
                encoder_out, hidden = self.encoder(hidden, x_frames[i])

        # decode
        decoder_out = self.decoder(hidden=hidden, x=encoder_out)
        return decoder_out

    def configure_optimizers(self):
        """ Configure the optimizer.

        Returns:
            torch.optim.Optimizer: Optimizer.
        """
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer
    
    def loss(self, output, labels):
        """ Calculate the loss.

        Args:
            output (torch.Tensor): Output of the model. Shape: (batch_size, sentence_length, label_size) i.e. (batch_size, 3, 19)
            labels (torch.Tensor): Labels of the data. Shape: (batch_size, sentence_length) i.e. (batch_size, 3)

        Returns:
            torch.Tensor: Loss.
        """
        loss = torch.zeros(1, device=output.device)
        for i in range(self.sentence_length):
            loss += self.loss_fn(output[:, i, :], labels[:, i])
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
        if self.use_joints:
            output = self(frames, joints)
        else:
            output = self(frames)
        loss = self.loss(output, labels)
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
    
    def validation_step(self, batch, batch_idx):
        """ Same as training step."""
        frames, joints, labels = batch
        if self.use_joints:
            output = self(frames, joints)
        else:
            output = self(frames)
        loss = self.loss(output, labels)
        self.log('val_loss', loss)
        self.calculate_accuracy(output, labels, train=False)
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
        self.reset_metrics_val()
        print_with_time(f"Epoch {self.current_epoch} val_acc: {epoch_acc}")

    def calculate_accuracy(self, output, labels, train=True):
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
        else:
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
        self.val_action_correct = 0
        self.val_color_correct = 0
        self.val_object_correct = 0
        self.val_total = 0