from pytorch_lightning import LightningModule
import torch.nn as nn
import torch
import numpy as np
from einops import rearrange
from helper import *


class LstmEncoder(LightningModule):
    """Encoder of the LSTM model. 
    Uses CNN to extract features from the input image, then feeds the features to the LSTM.

    Args:
        conv_features (list): List of convolutional features.
        input_size (int): Size of the input image.
        hidden_size (int): Size of the hidden state of the LSTM.
        num_layers (int, optional): Number of layers in the LSTM. Defaults to 1.

    Returns:
        torch.Tensor: Output of the LSTM.
        torch.Tensor: Hidden state of the LSTM.
    """

    def __init__(self, config):
        super().__init__()
        conv_features = config["convolution_layers_encoder"]
        dense_features = config["dense_layers_encoder"]
        self.hidden_size = config["lstm_hidden_size"]
        self.hidden_size = config["lstm_hidden_size"]
        self.num_layers = config["lstm_num_layers"]
        self.use_joints = config["use_joints"]
        self.num_joints = config["num_joints"]

        # Add convolutional layers
        conv_features.insert(0, 3) # Input should always be 3 (RGB)
        conv_layer_list = []
        for i in range(len(conv_features) - 1):
            conv_layer_list.append(nn.Conv2d(conv_features[i], conv_features[i + 1], kernel_size=3, stride=1, padding=1))
            conv_layer_list.append(nn.ReLU())
            conv_layer_list.append(nn.MaxPool2d(kernel_size=2, stride=2, padding=0))
         
        conv_layer_list.append(nn.Flatten(start_dim=1))
        self.conv_layers = nn.Sequential(*conv_layer_list)

        dense_layer_list = []
        dense_features.insert(0, self.num_joints)
        for i in range(len(dense_features) - 1):
            dense_layer_list.append(nn.Linear(dense_features[i], dense_features[i + 1]))
            dense_layer_list.append(nn.Sigmoid())
        self.dense_layers = nn.Sequential(*dense_layer_list)
        
        self.lstm = nn.LSTM(input_size=self.calculate_lstm_input_size(config), hidden_size=self.hidden_size,
                            num_layers=self.num_layers, batch_first=True)

    def forward(self, hidden, x_frames, x_joints=None):
        """Forward pass of the encoder.


        Args:
            x_input (torch.Tensor): Input image. Shape: (N, C, H, W) i.e. (batch_size, 3, 224, 398)
            hidden (torch.Tensor): Hidden state of the LSTM. Shape: (num_layers, N, hidden_size)

        Returns:
            torch.Tensor: Output of the LSTM. Shape: (N, hidden_size)
            torch.Tensor: Hidden state of the LSTM.
        """
        x = self.conv_layers(x_frames) # shape: (N, conv_features[-1] * H/2**len(conv_features) * W/2**len(conv_features))

        if self.use_joints:
            if x_joints is None:
                print_warning("Using joints but no joints given")
                x_joints = torch.zeros((x_frames.shape[0] ,self.num_joints), device=x_frames.device)
            dense_out = self.dense_layers(x_joints)
            x = torch.cat((x, dense_out), dim=1)

        x = rearrange(x, 'N C -> N 1 C')
        # Feed the features to the LSTM
        output, hidden = self.lstm(x , hidden)
        return output, hidden

    def init_hidden(self, batch_size, device):
        # Initialize the hidden state of the LSTM
        return  (torch.zeros((self.num_layers, batch_size, self.hidden_size), device=device),
                torch.zeros((self.num_layers, batch_size, self.hidden_size), device=device))
            
    def calculate_lstm_input_size(self, config):
        img_size = np.array((config["width"], config["height"]))
        for i in range(len(config["convolution_layers_encoder"]) - 1):
            img_size = img_size// 2
        total = int(np.prod(img_size) * config["convolution_layers_encoder"][-1])
        if config["use_joints"]:
            total += config["dense_layers_encoder"][-1]
        return total


class CnnDecoder(LightningModule):
    """Decoder of autoencoder.
    Feeds LSTM output to CNN to predict next frame.

    Args:
        conv_features (list): List of convolutional features.
        input_size (int): Size of the input image.
        hidden_size (int): Size of the hidden state of the LSTM.
        num_layers (int, optional): Number of layers in the LSTM. Defaults to 1.

    Returns:
        torch.Tensor: Predicted image.
    """

    def __init__(self, config):
        super().__init__()
        conv_features = config["convolution_layers_decoder"]
        dense_features = config["dense_layers_decoder"]
        input_shape = config["lstm_hidden_size"]
        output_shape = (config["height"], config["width"])
        self.num_joints = config["num_joints"]
        self.use_joints = config["use_joints"]
        
        # Build the layer list backwards then reverse it (to know the exact shapes)
        layer_list = []
        conv_features.insert(0, 1) # Input should always be 1 (unflattened from dense)
        current_shape = output_shape
        for i in range(len(conv_features) - 2,-1,-1):
            layer_list.append(nn.Upsample(size=current_shape, mode='bilinear'))
            layer_list.append(nn.ReLU())
            layer_list.append(nn.Conv2d(conv_features[i], conv_features[i + 1], kernel_size=3, stride=1, padding=1))
            current_shape = (current_shape[0] // 2, current_shape[1] // 2)
        layer_list.append(nn.Unflatten(dim=-1, unflattened_size=(current_shape)))
        layer_list.append(nn.Sigmoid())
        layer_list.append(nn.Linear(input_shape, np.prod(current_shape)))
        layer_list.reverse()

        layer_list.append(nn.Conv2d(conv_features[-1], 3, kernel_size=3, stride=1, padding=1))
        layer_list.append(nn.Sigmoid())
        self.conv_layers = nn.Sequential(*layer_list)

        dense_features.insert(0, input_shape)
        dense_list = []
        for i in range(len(dense_features) - 1):
            dense_list.append(nn.Linear(dense_features[i], dense_features[i + 1]))
            dense_list.append(nn.Sigmoid())
        dense_list.append(nn.Linear(dense_features[-1], self.num_joints))
        self.dense_layers = nn.Sequential(*dense_list)


    def forward(self, x):
        # x_input.shape : (N, 1, hidden_size)
        # output.shape : (N, L, 3, 224, 398)

        conv_out = self.conv_layers(x)
        if self.use_joints:
            dense_out = self.dense_layers(x)
            return conv_out, dense_out
        return conv_out


class LstmAutoencoder(LightningModule):
    """LSTM autoencoder model.

    Args:
        conv_features (list): List of convolutional features.
        input_size (int): Size of the input image.
        hidden_size (int): Size of the hidden state of the LSTM.
        num_layers (int, optional): Number of layers in the LSTM. Defaults to 1.

    Returns:
        torch.Tensor: Predicted image.
    """

    def __init__(self, config):
        super().__init__()
        self.learning_rate = config["learning_rate"]
        self.loss = nn.MSELoss()
        self.use_joints = config["use_joints"]
        self.num_joints = config["num_joints"]

        self.encoder = LstmEncoder(config)
        self.decoder = CnnDecoder(config)
        self.save_hyperparameters()

    def forward(self, x_frames, x_joints=None):
        """Forward pass of the autoencoder.

        Args:
            x_frames (torch.Tensor): Input frames. Shape: (N, L, C, H, W) i.e. (batch_size, sequence_length, 3, 224, 398)
            x_joints (torch.Tensor): Input joints. Shape: (N, L, J, 3) i.e. (batch_size, sequence_length, num_joints)

        Returns:
            torch.Tensor: Predicted frames. Shape: (N, L, C, H, W) i.e. (batch_size, sequence_length, 3, 224, 398)
            torch.Tensor: Predicted joints. Shape: (N, L, J)
        """
        hidden = self.encoder.init_hidden(x_frames.shape[0], device=x_frames.device) #init hidden state
        x_frames = rearrange(x_frames, 'n l c h w -> l n c h w')

        if self.use_joints:
            if x_joints is None:
                print_warning("Using joints but no joints given")
                x_joints = torch.zeros((x_frames.shape[0], x_frames.shape[1] ,self.num_joints), device=x_frames.device)
            else:
                x_joints = rearrange(x_joints, 'n l c -> l n c', c=self.num_joints)
            for i in range(x_frames.shape[0]):
                output_encoder, hidden = self.encoder(hidden, x_frames[i], x_joints[i])
        else:
            for i in range(x_frames.shape[0]):
                output_encoder, hidden = self.encoder(hidden, x_frames[i])    
        # output.shape : (N, hidden_size)

        # Feed the LSTM output to the decoder
        output_decoder = self.decoder(output_encoder) 
        return output_decoder

    def training_step(self, batch, batch_idx):
        loss = self.step(batch)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self.step(batch)
        self.log('val_loss', loss)
        return loss

    def step(self, batch):
        x_frames, x_joints, y_frames, y_joints = batch
         
        if self.use_joints:
            out_frames, out_joints = self(x_frames, x_joints)
            loss = self.loss(out_frames, y_frames)
            loss += self.loss(out_joints, y_joints)
        else:
            out_frames = self(x_frames)
            loss = self.loss(out_frames, y_frames)
        return loss


    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)
