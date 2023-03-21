import itertools
import torch
from torch import nn
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
        config (dict): Dictionary containing the configuration of the model.

    Attributes:
        hidden_size (int): Number of hidden units of the LSTM.
        num_layers (int): Number of layers of the LSTM.
        sentence_length (int): Length of the output sequence.
        label_size (int): Size of the label space.
        use_resnet (bool): Whether to use a ResNet first.
        flatten (nn.Flatten): Flatten layer.
        lstm (nn.LSTM): LSTM layer.
        linear (nn.Linear): Linear layer.
        dropout (nn.Dropout): Dropout layer.

    Methods:
        forward(x, output_size=None): Forward pass of the decoder.
        init_hidden(batch_size): Return zero vectors as initial hidden states.
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

    def forward(self, x, output_size=None):
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
        if output_size is not None:
            pred = torch.zeros((x.shape[0], output_size, self.label_size), device=x.device)
            range_ = range(output_size)
        else:
            pred = torch.zeros((x.shape[0], self.sentence_length, self.label_size), device=x.device) 
            range_ = range(self.sentence_length)
        for i in range_:
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

    Joints are not used in this model. 

    Args:
        config (dict): Dictionary containing the configuration parameters.
        encoder (LstmEncoder): Trained encoder part of the LstmAutoencoder model.

    Attributes:
        label_size (int): Size of the label space.
        learning_rate (float): Learning rate.
        sentence_length (int): Length of the output sequence.
        multi_sentence (bool): Whether the model should predict multiple sentences.
        dataset (str): Name of the dataset. 
        width (int): Width of the input images.
        height (int): Height of the input images.
        use_mask (bool): Whether to use a mask.
        use_augmentation (bool): Whether to use data augmentation.
        augmentation (DataAugmentation): Data augmentation module.
        encoder (LstmEncoder): Encoder part of the model.
        decoder (LstmDecoder): Decoder part of the model.

    Methods:
        forward(x, mask, output_size): Forward pass of the model.
        training_step(batch, batch_idx): Training step.
        validation_step(batch, batch_idx): Validation step.
        create_masks(): Create all possible masks.
        configure_optimizers(): Configure the optimizer.
        get_random_mask(batch_size, multi_sentence): Get a random mask.
        loss(pred, target, mask): Calculate the loss.
        training_epoch_end(outputs): Log training accuracy.
        validation_epoch_end(outputs): Log validation accuracy.
        calculate_accuracy(pred, target, train, gen): Calculate the accuracy.
        action_color_object_accuracy(pred, target, train, gen): Calculate the accuracy for the action-color-object task.
        action_color_material_object_accuracy(pred, target, train, gen): Calculate the accuracy for the action-color-material-object task.
        reset_metrics_train(): Reset the training metrics.
        reset_metrics_val(): Reset the validation metrics.
    """
    def __init__(self, config, encoder=None):
        super().__init__()
        self.save_hyperparameters()

        #self.use_joints = config["use_joints"]
        #self.num_joints = config["num_joints"]

        self.label_size = config["dictionary_size"]
        self.learning_rate = config["learning_rate"]
        self.sentence_length = config["sentence_length"]
        self.multi_sentence = config["multi_sentence"]
        self.dataset = config["dataset_name"]
        self.width = config["width"]
        self.height = config["height"]
        self.use_mask = config["use_mask"]
        self.use_augmentation = config["data_augmentation"]
        if self.use_augmentation:
            self.augmentation = DataAugmentation()

        if encoder is None:
            self.encoder = LstmEncoder(config)
        else:
            self.encoder = encoder
            self.encoder.freeze()
            self.encoder.requires_grad = False # Freeze the encoder 
        self.decoder = ClassificationLstmDecoder(config)
        if self.use_mask:
            self.create_masks()

        self.loss_fn = nn.CrossEntropyLoss()
        self.reset_metrics_train()
        self.reset_metrics_val()

    def create_masks(self):
        """ Create all possible masks for the frames.
        Length of each mask is sentence_length (3) and each position can be either 0 or 1.
        [0,0,0] is excluded because it means that no label is used.
        """
        self.masks = []
        for i in range(1, 2**self.sentence_length):
            # convert to binary
            mask = [int(x) for x in list('{0:0b}'.format(i))]
            # pad with zeros
            mask = [0] * (self.sentence_length - len(mask)) + mask
            self.masks.append(mask)
        self.masks = torch.tensor(self.masks, device=self.device)

        

    def get_random_mask(self, batch_size, multi_sentence_length=0):
        """ Get a random mask from the list of all possible masks.
        
        Each mask is the same for all inputs in a batch.
        """
        mask = self.masks[np.random.randint(len(self.masks))]
        if self.multi_sentence:
            # repeat mask for each sentence
            mask = mask.repeat((multi_sentence_length//self.sentence_length))
            # add 1 to the end of the mask to indicate the end of the sentence (EOS token)
            mask = torch.cat((mask, torch.ones(1)), dim=0)
        mask = mask.repeat(batch_size, 1)
        mask = mask.to(self.device) # don't know why this is necessary, but it is
        return mask
        
    def forward(self, x_frames, mask=None, multi_sentence_length=0):
        """ Forward pass of the model.

        Frames and masks are passed to the encoder.
        Encoder outputs a tensor for each frame, but only its last output is used.
        The last output is passed to the decoder.

        Args:
            x_frames (torch.Tensor): Tensor containing the frames of the video. Shape: (batch_size, num_frames, 3, height, width) i.e. (batch_size, num_frames, 3, 224, 398)
            mask (torch.Tensor): Tensor containing the masks. Shape: (batch_size, sentence_length) i.e. (batch_size, 3)
            multi_sentence_length (int): Length of the output sequence for the multi-sentence task.
        Returns:
            torch.Tensor: Output of the model. Shape: (batch_size, sentence_length, label_size) i.e. (batch_size, 3, 19)
        """
        # encode
        if self.use_mask:
            shortened_mask = mask[:, :self.sentence_length]
        else:
            shortened_mask = None
        encoder_out = self.encoder(x_frames, shortened_mask)
        encoder_out = encoder_out[-1]
        # decode
        if self.multi_sentence:
            return self.decoder(x=encoder_out, output_size=multi_sentence_length)
        return self.decoder(x=encoder_out)

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
    
    def loss(self, output, labels, mask=None):
        """ Calculate the loss.

        Args:
            output (torch.Tensor): Output of the model. Shape: (batch_size, sentence_length, label_size) i.e. (batch_size, 3, 19)
            labels (torch.Tensor): Labels of the data. Shape: (batch_size, sentence_length) i.e. (batch_size, 3)

        Returns:
            torch.Tensor: Loss.
        """
        batch_size = output.shape[0]
        loss = torch.zeros(batch_size, device=output.device)
        if mask is None:
            mask = torch.ones(batch_size, labels.shape[1], device=output.device)
        for i in range(labels.shape[1]):
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
        if self.dataset == 'ARC-GEN':
            frames, labels = batch
        else:
            frames, joints, labels = batch
        
        # random mask for each input
        if self.use_mask:
            mask = self.get_random_mask(frames.size(0), labels.size(1))
        else:
            mask = None

        if self.use_augmentation:
            augment_action = not mask[0][0]
            augment_color = not mask[0][1]
            if self.sentence_length == 3:
                augment_object = not mask[0][2]
                frames = self.augmentation(frames, augment_action, augment_color, augment_object)
            else:
                augment_material = not mask[0][2]
                augment_object = not mask[0][3]
                frames = self.augmentation(frames, augment_action, augment_color, augment_object, augment_material)

        if self.multi_sentence:
            output = self(frames, mask, multi_sentence_length=labels.shape[1])
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
        epoch_material_acc = self.training_material_correct * 100 / self.training_total
        epoch_object_acc = self.training_object_correct * 100 / self.training_total
        epoch_acc = (epoch_action_acc + epoch_color_acc + epoch_object_acc) / 3
        self.log('train_acc_action', epoch_action_acc, on_step=False, on_epoch=True)
        self.log('train_acc_color', epoch_color_acc, on_step=False, on_epoch=True)
        if self.sentence_length == 4:
            self.log('train_acc_material', epoch_material_acc, on_step=False, on_epoch=True)
        self.log('train_acc_object', epoch_object_acc, on_step=False, on_epoch=True)
        self.log('train_acc', epoch_acc, on_step=False, on_epoch=True)
        self.reset_metrics_train()
        print_with_time(f"Epoch {self.current_epoch} train acc: {epoch_acc}")
    
    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        """ Same as training step."""
        if self.dataset == 'ARC-GEN':
            frames, labels = batch
        else:
            frames, joints, labels = batch

        if self.multi_sentence and self.use_mask:
            mask = torch.ones(frames.size(0), labels.size(1), device=frames.device)
        elif self.use_mask:
            mask = torch.ones(frames.size(0), self.sentence_length, device=frames.device)
        else:
            mask = None

        if self.multi_sentence:
            output = self(frames, mask, multi_sentence_length=labels.shape[1])
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
        epoch_material_acc = self.val_material_correct * 100 / self.val_total
        epoch_object_acc = self.val_object_correct * 100 / self.val_total
        epoch_acc = (epoch_action_acc + epoch_color_acc + epoch_object_acc) / 3
        self.log('val_acc_action', epoch_action_acc, on_step=False, on_epoch=True)
        self.log('val_acc_color', epoch_color_acc, on_step=False, on_epoch=True)
        if self.sentence_length == 4:
            self.log('val_acc_material', epoch_material_acc, on_step=False, on_epoch=True)
        self.log('val_acc_object', epoch_object_acc, on_step=False, on_epoch=True)
        self.log('val_acc', epoch_acc, on_step=False, on_epoch=True)

        # generalization validation
        try:
            epoch_action_acc = self.val_gen_action_correct * 100 / self.val_gen_total
            epoch_color_acc = self.val_gen_color_correct * 100 / self.val_gen_total
            epoch_object_acc = self.val_gen_object_correct * 100 / self.val_gen_total
        except:
            epoch_action_acc = 0
            epoch_color_acc = 0
            epoch_object_acc = 0
        epoch_acc = (epoch_action_acc + epoch_color_acc + epoch_object_acc) / 3
        self.log('val_acc_action_gen', epoch_action_acc, on_step=False, on_epoch=True)
        self.log('val_acc_color_gen', epoch_color_acc, on_step=False, on_epoch=True)
        if self.sentence_length == 4:
            self.log('val_acc_material_gen', epoch_material_acc, on_step=False, on_epoch=True)
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
        if self.sentence_length == 3:
            self.action_color_object_accuracy(output, labels, train, gen)
        elif self.sentence_length == 4:
            self.action_color_material_object_accuracy(output, labels, train, gen)
        else:
            raise NotImplementedError
    
    def action_color_object_accuracy(self, output, labels, train=True, gen=False):
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

    def action_color_material_object_accuracy(self, output, labels, train=True, gen=False):
        """ Calculate the accuracy of the model.

        Args:
            output (torch.Tensor): Output of the model. Shape: (batch_size, sentence_length, label_size) i.e. (batch_size, 3, label_size)
            labels (torch.Tensor): Labels of the data. Shape: (batch_size, sentence_length)
            train (bool): Whether to calculate the training accuracy or the validation accuracy.
        """
        if self.multi_sentence:
            r = output.shape[1] // 4
        else:
            r = 1

        for i in range(r):
            _, action_output_batch = torch.max(output[:, i*4, :], dim=1)
            _, color_output_batch = torch.max(output[:, i*4+1, :], dim=1)
            _, material_output_batch = torch.max(output[:, i*4+2, :], dim=1)
            _, object_output_batch = torch.max(output[:, i*4+3, :], dim=1)

            if train:
                self.training_action_correct += torch.sum(action_output_batch == labels[:, i*4])
                self.training_color_correct += torch.sum(color_output_batch == labels[:, i*4+1])
                self.training_material_correct += torch.sum(material_output_batch == labels[:, i*4+2])
                self.training_object_correct += torch.sum(object_output_batch == labels[:, i*4+3])

                self.training_total += labels.shape[0]
            elif gen:
                # generalization validation
                self.val_gen_action_correct += torch.sum(action_output_batch == labels[:, i*4])
                self.val_gen_color_correct += torch.sum(color_output_batch == labels[:, i*4+1])
                self.val_gen_material_correct += torch.sum(material_output_batch == labels[:, i*4+2])
                self.val_gen_object_correct += torch.sum(object_output_batch == labels[:, i*4+3])

                self.val_gen_total += labels.shape[0]
            else:
                # normal validation
                self.val_action_correct += torch.sum(action_output_batch == labels[:, i*4])
                self.val_color_correct += torch.sum(color_output_batch == labels[:, i*4+1])
                self.val_material_correct += torch.sum(material_output_batch == labels[:, i*4+2])
                self.val_object_correct += torch.sum(object_output_batch == labels[:, i*4+3])

                self.val_total += labels.shape[0]

    def reset_metrics_train(self):
        """Reset metrics for training"""
        self.training_action_correct = 0
        self.training_color_correct = 0
        self.training_material_correct = 0
        self.training_object_correct = 0
        self.training_total = 0
    
    def reset_metrics_val(self):
        """Reset metrics for validation"""
        # normal validation
        self.val_action_correct = 0
        self.val_color_correct = 0
        self.val_material_correct = 0
        self.val_object_correct = 0
        self.val_total = 0
        # generalization validation
        self.val_gen_action_correct = 0
        self.val_gen_color_correct = 0
        self.val_gen_material_correct = 0
        self.val_gen_object_correct = 0
        self.val_gen_total = 0
