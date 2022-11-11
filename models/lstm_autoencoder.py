from pytorch_lightning import LightningModule
import torch.nn as nn
import torch
import numpy as np
from einops import rearrange
from helper import *
from models.conv_lstm_cell import *

class LstmEncoder(LightningModule):
    """Encoder of the LSTM model. 
    Uses ConvLSTM Cells to encode the input video.

    Args:
        config (dict): Dictionary containing the configuration of the model.

    Attributes:
        hidden_size (int): Hidden size of the LSTM.
        num_layers (int): Number of layers of the LSTM.
        use_joints (bool): Whether to use joints as input to the model.
        num_joints (int): Number of joints.
        conv_layers (nn.Sequential): Sequential model containing the convolutional layers.
        dense_layers (nn.Sequential): Sequential model containing the dense layers.
        lstm (nn.LSTM): LSTM model.
    """

    def __init__(self, config):
        super().__init__()
        convlstm_layers = config["convlstm_layers"] # e.g. [32,64,128]
        self.use_joints = config["use_joints"]
        in_chan = 3

        self.convlstm_1 = ConvLSTMCell(input_dim=in_chan,
                                               hidden_dim=convlstm_layers[0],
                                               kernel_size=(3, 3),
                                               bias=True)

        self.convlstm_2 = ConvLSTMCell(input_dim=convlstm_layers[0],
                                               hidden_dim=convlstm_layers[1],
                                               kernel_size=(3, 3),
                                               bias=True)

        self.convlstm_3 = ConvLSTMCell(input_dim=convlstm_layers[1],
                                                  hidden_dim=convlstm_layers[2],
                                                  kernel_size=(3, 3),
                                                  bias=True)

        self.maxpool = nn.MaxPool3d(kernel_size=(1, 2, 2))
        self.dropout = None
        if config["dropout"] > 0:
            self.dropout = nn.Dropout(p=config["dropout"])

    def forward(self, x, h_t, c_t, h_t2, c_t2, h_t3, c_t3):
        """Forward pass of the encoder.
        """
        if self.use_joints:
            #TODO: add the joints to the input
            print_warning("Joints are not yet implemented in the LSTM model.")
        seq_len = x.shape[1]

        for t in range(seq_len):
            h_t, c_t = self.convlstm_1(input_tensor=x[:, t, :, :],
                                               cur_state=[h_t, c_t])
            h_t_next = self.maxpool(h_t)
            if self.dropout:
                h_t_next = self.dropout(h_t_next)

            h_t2, c_t2 = self.convlstm_2(input_tensor=h_t_next,
                                                 cur_state=[h_t2, c_t2]) 
            h_t2_next = self.maxpool(h_t2)
            if self.dropout:
                h_t2_next = self.dropout(h_t2_next)

            h_t3, c_t3 = self.convlstm_3(input_tensor=h_t2_next,
                                                    cur_state=[h_t3, c_t3])
        h_t3_next = self.maxpool(h_t3)
        if self.dropout:
            h_t3_next = self.dropout(h_t3_next)

        return h_t3_next


class CnnDecoder(LightningModule):
    """Decoder of autoencoder.
    Feeds LSTM output to CNN to predict next frame.

    Args:
        config (dict): Dictionary containing the configuration of the model.
    
    Attributes:
        num_joints (int): Number of joints.
        use_joints (bool): Whether to use joints or not.
        conv_layers (nn.Sequential): Convolutional layers.
        dense_layers (nn.Sequential): Dense layers.
    """

    def __init__(self, config):
        super().__init__()
        conv_features = config["convolution_layers_decoder"]
        # TODO: implement joints branch
        input_shape = config["convlstm_features"]
        self.num_joints = config["num_joints"]
        self.use_joints = config["use_joints"]
        w = config["width"]
        h = config["height"]
        
        self.conv_layers = nn.Sequential(
            nn.Conv2d(input_shape, conv_features[0], kernel_size=3, stride=1, padding=1),
            nn.Upsample(size=(h//4,w//4), mode='nearest'),
            nn.ReLU(),
            nn.Conv2d(conv_features[0], 3, kernel_size=3, stride=1, padding=1),
            nn.Upsample(size=(h//2,w//2), mode='nearest'),
            nn.ReLU(),
            nn.Conv2d(conv_features[1], 3, kernel_size=3, stride=1, padding=1),
            nn.Upsample(size=(h, w), mode='nearest')
        )# TODO: dont hardcoder number of layers



    def forward(self, x):
        """Forward pass of the decoder.

        Args:
            x (torch.Tensor): Input of the decoder. Shape: (batch_size, 1, hidden_size)

        Returns:
            torch.Tensor: Output of the decoder. Shape: (batch_size, 3, height, width)
            torch.Tensor: Output of the decoder. Shape: (batch_size, num_joints) (only if use_joints is True)
        """
        conv_out = self.conv_layers(x)
        if self.use_joints:
            print_warning("Joints are not yet implemented in the CNN model.")
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
        self.save_hyperparameters(ignore=['encoder'])
        self.learning_rate = config["learning_rate"]
        self.loss = nn.MSELoss()
        self.use_joints = config["use_joints"]
        self.num_joints = config["num_joints"]
        self.init_length = config["init_length"]

        self.encoder = LstmEncoder(config)
        self.decoder = CnnDecoder(config)

    def forward(self, x):
        """Forward pass of the autoencoder.
        Given a frame and the previous hidden state, predicts the next frame.

        Args:
            x_frames (torch.Tensor): Input frames. Shape: (batch_size, 3, height, width) i.e. (batch_size, 3, 224, 398)
            x_joints (torch.Tensor): Input joints. Shape: (batch_size, num_joints) i.e. (batch_size, 6)

        Returns:
            torch.Tensor: Predicted frames. Shape: (batch_size, 3, height, width) i.e. (batch_size, 3, 224, 398)
            torch.Tensor: Predicted joints. Shape: (batch_size, num_joints) (Only if use_joints is True)
            torch.Tensor: Hidden state of the LSTM. Shape: (batch_size, hidden_size)
        """
        # find size of different input dimensions
        b, seq_len, _, h, w = x.size()

        # initialize hidden states
        h_t, c_t = self.encoder.convlstm_1.init_hidden(batch_size=b, image_size=(h, w))
        h_t2, c_t2 = self.encoder.convlstm_2.init_hidden(batch_size=b, image_size=(h//2, w//2))
        h_t3, c_t3 = self.encoder.convlstm_3.init_hidden(batch_size=b, image_size=(h//4, w//4))

        # autoencoder forward
        encoder_vector = self.encoder(x, h_t, c_t, h_t2, c_t2, h_t3, c_t3)
        output = self.decoder(encoder_vector)

        return output

    def training_step(self, batch, batch_idx):
        """Training step of the model. See step"""
        loss = self.step(batch)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        """Validation step of the model. See step"""
        loss = self.step(batch)
        self.log('val_loss', loss)
        return loss

    def step(self, batch):
        """Step of the model.

        Args:
            batch (tuple): Tuple of (frames, joints, labels)
                frames (torch.Tensor): Input frames. Shape: (batch_size, 3, height, width) i.e. (batch_size, 3, 224, 398)
                joints (torch.Tensor): Input joints. Shape: (batch_size, num_joints) i.e. (batch_size, 6)
                labels (torch.Tensor): Input labels. NOT USED.

        Returns:
            torch.Tensor: Loss of the model
        """
        x_frames, x_joints, _ = batch
        #TODO: add joints to the input
        out = self(x_frames[:, :-1, :, :, :])
        target = x_frames[:, -1, :, :, :]
        loss = self.loss(out, target)
        return loss
    
    def predict(self,batch):
        """Predicts the last frame given the all but the last frames.

        Args:
            batch (tuple): Tuple of (frames, joints, labels)
                frames (torch.Tensor): Input frames. Shape: (batch_size, 3, height, width) i.e. (batch_size, 3, 224, 398)
                joints (torch.Tensor): Input joints. Shape: (batch_size, num_joints) i.e. (batch_size, 6)
                labels (torch.Tensor): Input labels. NOT USED.

        Returns:
            torch.Tensor: Predicted frames. Shape: (batch_size, 3, height, width) i.e. (batch_size, 3, 224, 398)
            torch.Tensor: Predicted joints. Shape: (batch_size, num_joints) (Only if use_joints is True)
        """
        x_frames, x_joints, _ = batch
        return self(x_frames[:, :-1, :, :, :])
        

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)
