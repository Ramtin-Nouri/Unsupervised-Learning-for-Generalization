import torch
from torch import nn
from einops import rearrange
from collections import OrderedDict
from models.rnn_encoder_decoder import LstmEncoder, LstmDecoder
from models.torchvision_models import ResNet18
from pytorch_lightning import LightningModule


class EncoderDecoder(LightningModule):
    def __init__(self, config):
        """vision_architecture, pretrained_vision, seq2seq_architecture, dropout1=0.0, dropout2=0.0,
            image_features=256, hidden_dim=512, freeze=False, convolutional_features=1024,
            no_joints=False, dropout3=0.0):"""
        super().__init__()

        LABEL_SIZE = 19
        self.use_joints = config["use_joints"]
        JOINTS_SIZE = 6 if self.use_joints else 0

        # TODO: hardcode because will be replaced anyway:
        convolutional_features = 1024
        image_features = 256
        dropout1 = 0.0
        dropout2 = 0.0
        dropout3 = 0.0
        hidden_dim = 512
        freeze = False

        self.vision_model = ResNet18(pretrained=config["pretrained"],
                                        convolutional_features=convolutional_features, out_features=image_features,
                                        dropout1=dropout1, dropout2=dropout2, freeze=freeze)

        self.encoder = LstmEncoder(input_size=image_features + JOINTS_SIZE, hidden_size=hidden_dim)
        self.decoder = LstmDecoder(input_size=LABEL_SIZE, hidden_size=hidden_dim, dropout=dropout3)

        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, frames, joints):
        # input:
        # images shape : (N, L, 3, 224, 398)
        #   if precooked -> (N, L, conv_features)
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
        # TODO: dont hardcode
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
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
        return loss
    
    def validation_step(self, batch, batch_idx):
        frames, joints, labels = batch
        output = self(frames, joints)
        loss = self.loss(output, labels)
        self.log('val_loss', loss)
        return loss

