import torch
from torch import nn
from einops import rearrange
from collections import OrderedDict
from models.lstm_autoencoder import LstmEncoder
from pytorch_lightning import LightningModule
from helper import *


class ClassificationLstmDecoder(LightningModule):
    """ Decoder of the LSTM model for classification. """
    def __init__(self, input_size, hidden_size, num_layers=1, dropout=0.0):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.hidden = None
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size,
                            num_layers=num_layers, batch_first=True)
        self.dropout = nn.Dropout(p=dropout, inplace=False)
        self.linear = nn.Linear(hidden_size, input_size)  # can be replaced with proj size in LSTM

    def forward(self, x_input, encoder_hidden_states):
        lstm_out, self.hidden = self.lstm(x_input.unsqueeze(1), encoder_hidden_states)
        # lstm_out.shape    : (N, 1, hidden_size)
        output = self.linear(self.dropout(lstm_out.squeeze(1)))

        # output.shape : (N, 1, input_size)
        # self.hidden.shape : (1, N, hidden_size)
        return output, self.hidden

class LstmClassifier(LightningModule):
    """ LSTM model for classification. 

    Args:
        config (dict): Dictionary containing the configuration parameters.
        encoder (LstmEncoder): Trained encoder part of the LstmAutoencoder model.
    """
    def __init__(self, config, encoder):
        super().__init__()
        self.save_hyperparameters()

        LABEL_SIZE = 19
        self.use_joints = config["use_joints"]
        JOINTS_SIZE = 6 if self.use_joints else 0

        self.learning_rate = config["learning_rate"]
        # TODO: hardcode because will be replaced anyway:
        dropout = 0.0
        hidden_dim = 512

        self.encoder = encoder
        self.encoder.requires_frad = False
        self.decoder = ClassificationLstmDecoder(input_size=LABEL_SIZE, hidden_size=hidden_dim, dropout=dropout)

        self.loss_fn = nn.CrossEntropyLoss()
        self.reset_metrics_train()
        self.reset_metrics_val()
        
    def forward(self, frames, joints):
        # input:
        # images shape : (N, L, 3, 224, 398)
        # joints shape : (N, L, 6)
        #
        # ouput:
        # output shape : (N, Lout), Lout = 3

        N = frames.shape[0]  # batch size
        L = frames.shape[1]  # sequence length
        Lout = 3  # length of output sequence
        t = 19  # token size

        # forward pass
        frames = rearrange(frames, 'N L c w h -> (N L) c w h')

        frames_features = self.vision_model(frames)  # shape : (N L) feature_dim

        frames_features = rearrange(frames_features, '(N L) f -> N L f', N=N, L=L)

        if self.use_joints:
            sequence_input = torch.cat((frames_features, joints), dim=2)  # shape : (N, L, f+j)
        else:
            sequence_input = frames_features  # shape : (N, L, f)

        lstm_out, hidden = self.encoder(sequence_input)

        decoder_input = torch.zeros((N, t), device=frames.device)  # initial decoder input
        output = torch.zeros((N, Lout, t), device=frames.device)

        for i in range(Lout):
            # output shape : (N, input_size)
            decoder_output, hidden = self.decoder(decoder_input, hidden)
            output[:, i, :] = decoder_output
            decoder_input = decoder_output

        return output

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
        output = self(frames, joints)
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
        output = self(frames, joints)
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