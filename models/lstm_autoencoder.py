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
        conv_features = config["convolution_layers_encoder"]
        dense_features = config["dense_layers_encoder"]
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
            x_input (torch.Tensor): Input image. Shape: (batch_size, 3, height, width) i.e. (batch_size, 3, 224, 398)
            x_joints (torch.Tensor, optional): Input joints. Shape: (batch_size, num_joints) i.e. (batch_size, 6). Defaults to None.
            hidden (tuple): Tuple containing the hidden state (hx) and the cell state (cx) of the LSTM.
                hx (torch.Tensor): Hidden state of the LSTM. Shape: (num_layers, batch_size, hidden_size)
                cx (torch.Tensor): Cell state of the LSTM. Shape: (num_layers, batch_size, hidden_size)
        Returns:
            torch.Tensor: Output of the LSTM. Shape: (N, hidden_size)
            torch.Tensor: Hidden state of the LSTM. (hx, cx)
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
        """Initialize the hidden state of the LSTM.

        Args:
            batch_size (int): Batch size of the input.
            device (torch.device): Device on which the input is located.

        Returns:
            tuple: Tuple containing the hidden state (hx) and the cell state (cx) of the LSTM.
                hx (torch.Tensor): Hidden state of the LSTM. Shape: (num_layers, batch_size, hidden_size)
                cx (torch.Tensor): Cell state of the LSTM. Shape: (num_layers, batch_size, hidden_size)
        """
        return  (torch.zeros((self.num_layers, batch_size, self.hidden_size), device=device),
                torch.zeros((self.num_layers, batch_size, self.hidden_size), device=device))
            
    def calculate_lstm_input_size(self, config):
        """Calculate the input size of the LSTM.
        Start with the input size of the image, then half the size for each maxpooling layer.
        Then multiply with the number of features in the last convolutional layer.

        Args:
            config (dict): Dictionary containing the configuration of the model.

        Returns:
            int: Input size of the LSTM.
        """
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
        """Forward pass of the decoder.

        Args:
            x (torch.Tensor): Input of the decoder. Shape: (batch_size, 1, hidden_size)

        Returns:
            torch.Tensor: Output of the decoder. Shape: (batch_size, 3, height, width)
            torch.Tensor: Output of the decoder. Shape: (batch_size, num_joints) (only if use_joints is True)
        """
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
        self.save_hyperparameters(ignore=['encoder'])
        self.learning_rate = config["learning_rate"]
        self.loss = nn.MSELoss()
        self.use_joints = config["use_joints"]
        self.num_joints = config["num_joints"]
        self.init_length = config["init_length"]

        self.encoder = LstmEncoder(config)
        self.decoder = CnnDecoder(config)

    def forward(self, x_frames, x_joints=None, hidden=None):
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
        if hidden is None:
            print_warning("No hidden state given, initializing hidden state")
            hidden = self.encoder.init_hidden(x_frames.shape[0], x_frames.device)

        if self.use_joints:
            if x_joints is None:
                print_warning("Using joints but no joints given")
                x_joints = torch.zeros((x_frames.shape[0], x_frames.shape[1] ,self.num_joints), device=x_frames.device)
            output_encoder, hidden = self.encoder(hidden, x_frames, x_joints)
        else:
            output_encoder, hidden = self.encoder(hidden, x_frames)
        # output.shape : (N, hidden_size)

        # Feed the LSTM output to the decoder
        output_decoder = self.decoder(output_encoder) 
        return output_decoder, hidden

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
        x_frames = rearrange(x_frames, 'n l c h w -> l n c h w')
        x_joints = rearrange(x_joints, 'n l c -> l n c', c=self.num_joints)
        hidden = self.encoder.init_hidden(x_frames.shape[1], device=x_frames.device) #init hidden state
        
        # feed starting frames to the model
        start_frames = x_frames[:self.init_length]
        start_joints = x_joints[:self.init_length]
        for i in range(self.init_length):
            if self.use_joints: 
                _, hidden = self.encoder(hidden, start_frames[i], start_joints[i])
            else:
                _, hidden = self.encoder(hidden, start_frames[i])

        # feed the rest of the frames to the model
        x_frames = x_frames[self.init_length:]
        x_joints = x_joints[self.init_length:]
        losses = []
        for i in range(len(x_frames)-1):
            if self.use_joints:
                (output_frames, output_joints), hidden = self(x_frames[i], x_joints[i], hidden)
                losses.append(self.loss(output_frames, x_frames[i+1]))
                losses.append(self.loss(output_joints, x_joints[i+1]))
            else:
                output_frames, hidden = self(x_frames[i], hidden=hidden)
                losses.append(self.loss(output_frames, x_frames[i+1]))
        return torch.sum(torch.stack(losses))
    
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
        x_frames = rearrange(x_frames, 'n l c h w -> l n c h w')
        x_joints = rearrange(x_joints, 'n l c -> l n c', c=self.num_joints)
        hidden = self.encoder.init_hidden(x_frames.shape[1], device=x_frames.device)

        for i in range(len(x_frames)-1):
            if self.use_joints:
                (output_frames, output_joints), hidden = self(x_frames[i], x_joints[i], hidden)
            else:
                output_frames, hidden = self(x_frames[i], hidden=hidden)
        
        if self.use_joints:
            return output_frames, output_joints
        else:
            return output_frames        
        

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)
