import torch
from torch import nn
from einops import rearrange
from collections import OrderedDict
from models.lstm_autoencoder import LstmEncoder
from pytorch_lightning import LightningModule
from helper import *


class ClassificationLstmDecoder(LightningModule):
    """ Decoder of the LSTM model for classification. """
    def __init__(self, config, output_size):
        super().__init__()
        self.hidden_size = config["lstm_hidden_size"]
        self.num_layers=config["lstm_num_layers"]
        self.sentence_length = 3 # TODO: dont hardcode
        self.label_size = 19 #TODO: remove hard coded value

        self.lstm = nn.LSTM(input_size=self.hidden_size, hidden_size=self.hidden_size,
                            num_layers=self.num_layers, batch_first=True)
        self.linear = nn.Linear(self.hidden_size, output_size)

    def forward(self, hidden, x):
        pred = torch.zeros((x.shape[0], self.sentence_length, self.label_size), device=x.device)
        for i in range(self.sentence_length):
            lstm_out, hidden = self.lstm(x, hidden)
            linear_out = self.linear(lstm_out)
            pred[:, i, :] = linear_out.squeeze(1)
            
        return pred
        
    def init_hidden(self, batch_size, device):
        # Initialize the hidden state of the LSTM
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

        self.label_size = 19 #TODO: remove hard coded value
        self.use_joints = config["use_joints"]
        self.learning_rate = config["learning_rate"]
        self.num_joints = config["num_joints"]

        self.encoder = encoder
        self.encoder.requires_frad = False
        self.decoder = ClassificationLstmDecoder(output_size=self.label_size, config=config)

        self.loss_fn = nn.CrossEntropyLoss()
        self.reset_metrics_train()
        self.reset_metrics_val()
        
    def forward(self, x_frames, x_joints=None):
        """ Forward pass of the model.

        Args:
            x_frames (torch.Tensor): Tensor containing the frames of the video. Shape: (batch_size, num_frames, 3, 224, 398)
            x_joints (torch.Tensor): Tensor containing the joints of the video. Shape: (batch_size, num_frames, num_joints, 3)

        Returns:
            torch.Tensor: Output of the model. Shape: (batch_size, Lout, label_size) i.e. (batch_size, 3, label_size)
        """
        N = x_frames.shape[0]  # batch size
        L = x_frames.shape[1]  # sequence length

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
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer
    
    def loss(self, output, labels):
        loss = self.loss_fn(output[:, 0, :], labels[:, 0])
        loss += self.loss_fn(output[:, 1, :], labels[:, 1])
        loss += self.loss_fn(output[:, 2, :], labels[:, 2])
        return loss

    def training_step(self, batch, batch_idx):
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
        epoch_action_acc = self.training_action_correct * 100 / self.training_total
        epoch_color_acc = self.training_color_correct * 100 / self.training_total
        epoch_object_acc = self.training_object_correct * 100 / self.training_total
        epoch_acc = (self.training_action_correct + self.training_color_correct + self.training_object_correct) / 3
        self.log('train_acc_action', epoch_action_acc, on_step=False, on_epoch=True)
        self.log('train_acc_color', epoch_color_acc, on_step=False, on_epoch=True)
        self.log('train_acc_object', epoch_object_acc, on_step=False, on_epoch=True)
        self.log('train_acc', epoch_acc, on_step=False, on_epoch=True)
        self.reset_metrics_train()
        print_with_time(f"Epoch {self.current_epoch} train acc: {epoch_acc}")
    
    def validation_step(self, batch, batch_idx):
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
        self.training_action_correct = 0
        self.training_color_correct = 0
        self.training_object_correct = 0
        self.training_total = 0
    
    def reset_metrics_val(self):
        self.val_action_correct = 0
        self.val_color_correct = 0
        self.val_object_correct = 0
        self.val_total = 0