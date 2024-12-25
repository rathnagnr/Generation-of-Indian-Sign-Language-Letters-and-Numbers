# import requirements
import torch                                                        #type:ignore
from torch import nn                                                #type:ignore


# Define self-attention:
class Self_Attention(nn.Module):
    """ Self-Attention Layer """
    def __init__(self, in_dim, activation):
        """
        Initializes the self-attention module.

        Args:
            in_dim (int): Number of input channels.
            activation: Activation function to be used (currently not utilized).
        """
        super(Self_Attention, self).__init__()
        self.chanel_in = in_dim
        self.activation = activation
        # Query, Key, and Value convolutions with reduced channel dimensions for Q and K
        self.query_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        # Trainable scaling factor for attention output
        self.gamma = nn.Parameter(torch.zeros(1))
        # Softmax layer for calculating attention weights
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        """
        Forward pass for the self-attention module.

        Args:
            x (tensor): Input feature map with shape (B, C, W, H), where
                        B is batch size, C is number of channels,
                        W is width, and H is height.

        Returns:
            out (tensor): Output feature map after applying self-attention, 
                          with shape (B, C, W, H).
            attention (tensor): Attention map with shape (B, N, N), where
                                N = W * H (spatial locations).
        """
        # Extract dimensions of input feature map
        m_batchsize, C, width, height = x.size()
        # Generate query projection: shape (B, N, C') where N = W * H, C' = in_dim // 8
        proj_query = self.query_conv(x).view(m_batchsize, -1, width * height).permute(0, 2, 1)
        # Generate key projection: shape (B, C', N)
        proj_key = self.key_conv(x).view(m_batchsize, -1, width * height)
        # Compute energy (similarity) matrix: shape (B, N, N)
        energy = torch.bmm(proj_query, proj_key)
        # Compute attention weights: shape (B, N, N)
        attention = self.softmax(energy)
        # Generate value projection: shape (B, C, N)
        proj_value = self.value_conv(x).view(m_batchsize, -1, width * height)
        # Compute weighted sum of values: shape (B, C, W, H)
        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(m_batchsize, C, width, height)
        # Combine attention output with the original input using scaling factor gamma
        out = self.gamma * out + x

        return out
