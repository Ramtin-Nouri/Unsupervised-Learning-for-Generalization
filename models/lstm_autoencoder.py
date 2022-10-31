from turtle import width
from pytorch_lightning import LightningModule
import torch.nn as nn
import torch
import numpy as np
from einops import rearrange


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

    def __init__(self, conv_features, input_size, hidden_size, num_layers=1):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # Add convolutional layers
        layer_list = []
        for i in range(len(conv_features) - 1):
            layer_list.append(nn.Conv2d(conv_features[i], conv_features[i + 1], kernel_size=3, stride=1, padding=0))
            layer_list.append(nn.ReLU())
            layer_list.append(nn.MaxPool2d(kernel_size=2, stride=2, padding=0))
        layer_list.append(nn.Flatten())
        self.conv_layers = nn.Sequential(*layer_list)
        
        self.hidden = None
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size,
                            num_layers=num_layers, batch_first=True)

    def forward(self, x_input):
        # x_input.shape : (N, L, 3, 224, 398)
        # self.hidden.shape : (1, N, hidden_size)
        # output.shape : (N, L, hidden_size)

        # Extract features from the input image
        x_input = self.conv_layers(x_input)
        # x_input.shape : (N, L, conv_features[-1])

        # Feed the features to the LSTM
        output, self.hidden = self.lstm(x_input, self.hidden)
        return output, self.hidden

    def init_hidden(self, batch_size):
        # Initialize the hidden state of the LSTM
        self.hidden =  (torch.zeros(self.num_layers, batch_size, self.hidden_size),
                torch.zeros(self.num_layers, batch_size, self.hidden_size))


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

    def __init__(self, conv_features):
        super().__init__()
        # Add convolutional layers
        conv_features = conv_features
        layer_list = []
        for i in range(len(conv_features) - 1):
            layer_list.append(nn.Conv2d(conv_features[i], conv_features[i + 1], kernel_size=3, stride=1, padding=0))
            layer_list.append(nn.ReLU())
            layer_list.append(nn.Upsample(scale_factor=2))
        layer_list.append(nn.Conv2d(conv_features[-1], 3, kernel_size=3, stride=1, padding=0))
        layer_list.append(nn.Sigmoid())
        self.conv_layers = nn.Sequential(*layer_list)

    def forward(self, x_input):
        # x_input.shape : (N, L, hidden_size)
        # output.shape : (N, L, 3, 224, 398)

        # Feed the LSTM output to the CNN
        output = self.conv_layers(x_input)
        return output


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
        self.config = config
        self.learning_rate = config["learning_rate"]


        lstm_in = self.calculate_lstm_input_size()
        self.encoder = LstmEncoder(config["convolution_layers_encoder"], lstm_in, config["lstm_hidden_size"], config["lstm_num_layers"])
        self.decoder = CnnDecoder(config["convolution_layers_decoder"])

    def forward(self, x_input):
        # x_input.shape : (N, L, 3, 224, 398)
        # output.shape : (N, L, 3, 224, 398)

        # Feed the input to the encoder
        self.encoder.init_hidden(x_input.shape[0]) #init hidden state

        frames = rearrange(x_input, 'n l c h w -> (n l) c h w')
        frames = self.encoder.conv_layers(frames)
        frames = rearrange(frames, '(n l) c h w -> n l c h w', l=x_input.shape[1])
        output, _ = self.encoder.lstm(frames, self.encoder.hidden)

        # Feed the output of the encoder to the decoder
        output = self.decoder(output)
        return output

        for i in range(x_input.shape[1]):
            output, _ = self.encoder(x_input[:, i, :, :, :])
        # output.shape : (N, L, hidden_size)

        # Feed the LSTM output to the decoder
        output = self.decoder(x_input)
        return output

    def training_step(self, batch, batch_idx):
        x, _, y, _ = batch #TODO: INCORPARATE JOINTS
        y_hat = self(x)
        loss = nn.MSELoss()(y_hat, y)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, _, y, _ = batch
        y_hat = self(x)
        loss = nn.MSELoss()(y_hat, y)
        self.log('val_loss', loss)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)

    def calculate_lstm_input_size(self):
        img_size = np.array((self.config["width"], self.config["height"]))
        for i in range(len(self.config["convolution_layers_encoder"]) - 1):
            img_size = img_size// 2
        return int(np.prod(img_size) * self.config["convolution_layers_encoder"][-1])
