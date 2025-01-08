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


# Define weight standardised convolution layer
class WSConv2d(nn.Module):
    def __init__(
        self, in_channels, out_channels, kernel_size=3, stride=1, padding=1,
    ):
        super(WSConv2d, self).__init__()
        self.conv      = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.scale     = (2 / (in_channels * (kernel_size ** 2))) ** 0.5
        self.bias      = self.conv.bias #Copy the bias of the current column layer
        self.conv.bias = None      #Remove the bias
        # initialize conv layer
        nn.init.normal_(self.conv.weight)
        nn.init.zeros_(self.bias)
    # Perform forward pass
    def forward(self, x):
        return self.conv(x * self.scale) + self.bias.view(1, self.bias.shape[0], 1, 1)
    
# Define data normalization(L2)
class PixelNorm(nn.Module):
    def __init__(self):
        super(PixelNorm, self).__init__()
        # Take the epsilon a samll value
        self.epsilon = 1e-8
    def forward(self, x):
        return x / torch.sqrt(torch.mean(x ** 2, dim=1, keepdim=True) + self.epsilon)
    

# Define a convolution block that uses pixel normalization and two weight-standardized convolutions.
class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, use_pixelnorm=True):
        """
        Initializes the convolution block.
        Args:
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels.
            use_pixelnorm (bool): Flag to determine whether to apply PixelNorm. Default is True.
        """
        super(ConvBlock, self).__init__()
        self.use_pn = use_pixelnorm  # Store the pixel normalization flag
        # First convolution layer with weight standardization
        self.conv1 = WSConv2d(in_channels, out_channels)
        # Add self-attention layers
        self.attention1 = Self_Attention(256, 'relu')  # First attention layer for specific conditions
        self.attention2 = Self_Attention(128, 'relu')  # Second attention layer (currently unused)
        # Second convolution layer with weight standardization
        self.conv2 = WSConv2d(out_channels, out_channels)
        # LeakyReLU activation function
        self.leaky = nn.LeakyReLU(0.2)
        # Pixel normalization layer
        self.pn = PixelNorm()

    def forward(self, x, step):
        """
        Forward pass for the convolution block.
        Args:
            x (tensor): Input feature map.
            step (int): Step value to control the application of attention layers.
        Returns:
            x (tensor): Output feature map after applying the convolution block.
        """
        # Apply first convolution and activation
        x = self.leaky(self.conv1(x))
        # Apply pixel normalization if enabled
        x = self.pn(x) if self.use_pn else x
        # Apply the first self-attention layer only if step == 3
        if step == 3:
            x = self.attention1(x)
        # Uncomment the following lines if you want to enable the second attention layer for step == 4
        # elif step == 4:
        #     x = self.attention2(x)

        # Apply second convolution and activation
        x = self.leaky(self.conv2(x))
        # Apply pixel normalization if enabled
        x = self.pn(x) if self.use_pn else x
        return x