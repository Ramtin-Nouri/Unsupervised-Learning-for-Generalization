import torch
from torch import nn
from einops import rearrange
from collections import OrderedDict
from pytorch_lightning import LightningModule
from helper import *
from models.conv_lstm_cell import *
from torchvision.models import resnet18
from torchvision.transforms import *
import numpy as np


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
            convlstm = ConvLSTMCell(input_dim=convlstm_layers[i-1]+mask_channels,
                                               hidden_dim=convlstm_layers[i],
                                               kernel_size=(3, 3),
                                               bias=True)
            self.convLSTMs.append(convlstm)
        self.convLSTMs = nn.ModuleList(self.convLSTMs)


        self.maxpool = nn.MaxPool3d(kernel_size=(1, 2, 2))

    def forward(self, x, mask, h_t, c_t):
        """Forward pass of the encoder.

        Args:
            x (torch.Tensor): Input video. Shape: (batch_size, seq_len, channels, height, width).
            mask (torch.Tensor): Mask of the input video.
            h_t (torch.Tensor): Hidden state of the LSTM.
            c_t (torch.Tensor): Cell state of the LSTM.
        """
        if self.use_joints:
            #TODO: add the joints to the input
            print_warning("Joints are not yet implemented in the LSTM model.")
        seq_len = x.shape[1]

        for t in range(seq_len):
            x_t = x[:, t, :, :, :]
            if self.use_resnet:
                with torch.no_grad():
                    x_t = self.vision_pre_model(x_t)

            for i in range(len(self.convLSTMs)):
                #add the mask to the input
                mask_expanded = mask.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, x_t.shape[2], x_t.shape[3])
                x_t = torch.cat((x_t, mask_expanded), dim=1)
                h_t[i], c_t[i] = self.convLSTMs[i](x_t, (h_t[i], c_t[i]))
                x_t = self.maxpool(h_t[i])

        return x_t

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

class LstmClassifier(LightningModule):
    """ LSTM model for classification. 

    Args:
        config (dict): Dictionary containing the configuration parameters.
        encoder (LstmEncoder): Trained encoder part of the LstmAutoencoder model.
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

        self.encoder = LstmEncoder(config)
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

        # decode
        decoder_out = self.decoder(x=encoder_out)
        return decoder_out

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

        Returns:
            torch.Tensor: Loss.
        """
        loss = torch.zeros(1, device=output.device)
        for i in range(self.sentence_length):
            loss += self.loss_fn(output[:, i, :], labels[:, i]) * mask[:, i]
        
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
        
        # random mask for each input
        mask = [self.masks[np.random.choice(len(self.masks))]] * frames.size(0) #each batch has the same mask
        mask = torch.tensor(mask, device=frames.device)

        if self.use_augmentation:
            transformations = []
            transformations.append(RandomRotation(degrees=(0, 25)))
            if not mask[0,0]:
                # if masking action, randomly flip
                transformations.append(RandomHorizontalFlip(p=0.3))
            
            if not mask[0,1]:
                # if masking color, randomly change color
                transformations.append(ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.5))
                transformations.append(RandomGrayscale(p=0.1))

            if not mask[0,2]:
                # if masking object, randomly blur
                transformations.append(GaussianBlur(kernel_size=3, sigma=(0.1, 3)))

            # apply transformations
            compose = Compose(transformations)
            for i in range(frames.size(0)):
                frames[i] = compose(frames[i])

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
    
    def validation_step(self, batch, batch_idx):
        """ Same as training step."""
        frames, joints, labels = batch
        mask = torch.ones(frames.size(0), 3, device=frames.device)

        if self.use_joints:
            output = self(frames, mask, joints)
        else:
            output = self(frames, mask)
        loss = self.loss(output, labels, mask)
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
