from pytorch_lightning import LightningModule
import torch.nn as nn
import torch
import numpy as np
from einops import rearrange
from torchvision.models import resnet18
from helper import *
from models.conv_lstm_cell import *
from data_augmentation import DataAugmentation

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
        self.height = config["height"]
        self.width = config["width"]
        mask_channels = 3
        self.use_resnet = config["use_resnet"]
        if self.use_resnet:
            in_chan = 256
            self.height = int(round(self.height/16))
            self.width = int(round(self.width/16))
            self.vision_pre_model = get_layers_until(resnet18(pretrained=True), "layer3")
        else:
            in_chan = 3
        in_chan += mask_channels


        convlstm_1 = ConvLSTMCell(input_dim=in_chan,
                                               hidden_dim=convlstm_layers[0],
                                               kernel_size=(3, 3),
                                               bias=True)

        self.convLSTMs = []
        self.convLSTMs.append(convlstm_1)
        for i in range(1, len(convlstm_layers)):
            convlstm = ConvLSTMCell(input_dim=convlstm_layers[i-1],
                                               hidden_dim=convlstm_layers[i],
                                               kernel_size=(3, 3),
                                               bias=True)
            self.convLSTMs.append(convlstm)
        self.convLSTMs = nn.ModuleList(self.convLSTMs)


        self.maxpool = nn.MaxPool3d(kernel_size=(1, 2, 2))

    def forward(self, x, mask, h_t, c_t):
        """Forward pass of the encoder.
        """
        if self.use_joints:
            #TODO: add the joints to the input
            print_warning("Joints are not yet implemented in the LSTM model.")
        seq_len = x.shape[1]

        outputs = []
        for t in range(seq_len):
            x_t = x[:, t, :, :, :]
            #add the mask to the input
            mask_expanded = mask.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, x_t.shape[2], x_t.shape[3])
            x_t = torch.cat((x_t, mask_expanded), dim=1)
            for i in range(len(self.convLSTMs)):
                h_t[i], c_t[i] = self.convLSTMs[i](x_t, (h_t[i], c_t[i]))
                x_t = self.maxpool(h_t[i])
            outputs.append(x_t)

        return outputs

    def init_hidden(self, batch_size):
        """Initializes the hidden state of the LSTM.
        Args:
            batch_size (int): Batch size.
        Returns:
            array of torch.Tensor: Hidden states of the LSTM.
            array of torch.Tensor: Cell states of the LSTM.
        """
        # initialize hidden states
        h_t = []
        c_t = []
        for i in range(len(self.convLSTMs)):
            new_h, new_c = self.convLSTMs[i].init_hidden(batch_size, image_size=(self.height//2**i, self.width//2**i))
            h_t.append(new_h)
            c_t.append(new_c)
        return h_t, c_t


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
        input_shape = config["convlstm_layers"][-1]
        self.num_joints = config["num_joints"]
        self.use_joints = config["use_joints"]
        w = config["width"]
        h = config["height"]

        conv_layers = []
        for i in range(len(conv_features)):
            conv_layers.append(nn.Conv2d(input_shape, conv_features[i], kernel_size=3, padding=1))
            upsample_w = w // (2 ** (len(conv_features)-(i+1)) )
            upsample_h = h // (2 ** (len(conv_features)-(i+1)) )
            conv_layers.append(nn.Upsample(size=(upsample_h, upsample_w), mode='nearest'))
            conv_layers.append(nn.ReLU())
            # nn.BatchNorm2d(conv_features[i]) # Use batchnorm?
            input_shape = conv_features[i]
        # Last layer without ReLU
        conv_layers.append(nn.Conv2d(input_shape, 3, kernel_size=3, padding=1))

        self.conv_layers = nn.Sequential(*conv_layers)

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
        self.predict_ahead = config["predict_ahead"]

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
        mask = torch.ones(b, 3, device=x.device)

        # initialize hidden states
        h_t = []
        c_t = []
        for i in range(len(self.encoder.convLSTMs)):
            new_h, new_c = self.encoder.convLSTMs[i].init_hidden(b, image_size=(h//2**i, w//2**i))
            h_t.append(new_h)
            c_t.append(new_c)
            

        # autoencoder forward
        encoder_vectors = self.encoder(x, mask, h_t, c_t)
        outputs = []
        for vec in encoder_vectors:
            outputs.append(self.decoder(vec))
        return torch.stack(outputs, dim=1)

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
        # TODO: add joints to the input

        out = self(x_frames[:, :-self.predict_ahead]) # feed all frames except the last one
        target = x_frames[:, self.predict_ahead:] # target is the next frame respectively
        loss = self.loss(out[:,self.init_length:], target[:,self.init_length:]) # only calculate loss for the frames after the initialization
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
        return self(x_frames[:, :-self.predict_ahead, :, :, :])[:,-1]
        

    def configure_optimizers(self):
        """Configures the optimizer.

        Returns:
            torch.optim.Adam: Adam optimizer
        """
        # lr scheduler
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=2, verbose=True)
        return {
            'optimizer': optimizer,
            'lr_scheduler': scheduler,
            'monitor': 'val_loss'
        }

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
