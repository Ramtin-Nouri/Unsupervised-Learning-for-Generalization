"""
    Convolutional LSTM Cell
    Pretty much the same as: https://github.com/holmdk/Video-Prediction-using-PyTorch
"""
import torch.nn as nn
import torch
from pytorch_lightning import LightningModule


class ConvLSTMCell(LightningModule):
    """Convolutional LSTM cell.
    See: https://arxiv.org/pdf/1506.04214v2.pdf

    Args:
        input_dim (int): Number of channels of the input tensor.
        hidden_dim (int): Number of channels of the hidden state.
        kernel_size (tuple): Size of the convolutional kernel.
        bias (bool): Whether to use bias in the convolutional layers.

    Attributes:
        conv (nn.Conv2d): Convolutional layer.
        hidden_dim (int): Number of channels of the hidden state.
    """

    def __init__(self, input_dim, hidden_dim, kernel_size, bias):
        super(ConvLSTMCell, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        self.kernel_size = kernel_size
        self.padding = kernel_size[0] // 2, kernel_size[1] // 2
        self.bias = bias

        self.conv = nn.Conv2d(in_channels=self.input_dim + self.hidden_dim,
                              out_channels=4 * self.hidden_dim,
                              kernel_size=self.kernel_size,
                              padding=self.padding,
                              bias=self.bias).to(device=self.device)

    def forward(self, input_tensor, cur_state):
        """Forward pass of the ConvLSTM cell.

        Args:
            input_tensor (torch.Tensor): Input tensor of shape (batch_size, seq_len, input_dim, height, width).
            cur_state (list): List containing the current hidden state and the current cell state.
        """
        h_cur, c_cur = cur_state

        # concatenate along channel axis
        combined = torch.cat([input_tensor, h_cur], dim=1)

        combined_conv = self.conv(combined)
        cc_i, cc_f, cc_o, cc_g = torch.split(
            combined_conv, self.hidden_dim, dim=1)
        i = torch.sigmoid(cc_i)
        f = torch.sigmoid(cc_f)
        o = torch.sigmoid(cc_o)
        g = torch.tanh(cc_g)

        c_next = f * c_cur + i * g
        h_next = o * torch.tanh(c_next)

        return h_next, c_next

    def init_hidden(self, batch_size, image_size):
        """Return zero vectors as initial hidden states.

        Args:
            batch_size (int): Batch size.
            image_size (tuple): Size of the image.

        Returns:
            torch.Tensor: Initial hidden state.
        """
        height, width = image_size
        return (torch.zeros(batch_size, self.hidden_dim, height, width, device=self.conv.weight.device),
                torch.zeros(batch_size, self.hidden_dim, height, width, device=self.conv.weight.device))
