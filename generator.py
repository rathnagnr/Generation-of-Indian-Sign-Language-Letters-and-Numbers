# import requirements
import torch                                                  #type:ignore
from torch import nn                                          #type:ignore
import torch.nn.functional as F                               #type:ignore
#from train import factors, device
from attention import PixelNorm, WSConv2d, ConvBlock

factors = [1, 1, 1, 1, 1 / 2, 1 / 4, 1 / 8, 1 / 16, 1 / 32]
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Code of Generator model
class Generator(nn.Module):
    def __init__(self, z_dim, in_channels, num_classes, img_channels=3):
        """
        Initializes the Generator model.
        
        Parameters:
        - z_dim: Dimension of the input noise vector.
        - in_channels: Number of channels in the intermediate feature maps.
        - num_classes: Number of distinct classes for conditional generation.
        - img_channels: Number of channels in the output image (default: 3 for RGB images).
        """
        super(Generator, self).__init__()

        # Embedding layer for class conditional generation
        self.class_embedding = nn.Embedding(num_classes, num_classes)

        # Initial block of the generator
        self.initial = nn.Sequential(
            PixelNorm(),  # Normalizes the input feature map
            nn.ConvTranspose2d(z_dim + num_classes, in_channels, 4, 1, 0),  # Deconvolution layer
            nn.LeakyReLU(0.2),  # Activation function
            WSConv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1),  # Weighted Standardized Conv2D
            nn.LeakyReLU(0.2),  # Activation function
            PixelNorm(),  # Normalizes the feature map
        )

        # Initial RGB layer for generating the first output image
        self.initial_rgb = WSConv2d(
            in_channels, img_channels, kernel_size=1, stride=1, padding=0
        )

        # Lists to store progressive blocks and corresponding RGB layers
        self.prog_blocks, self.rgb_layers = (
            nn.ModuleList([]),
            nn.ModuleList([self.initial_rgb]),
        )

        # Adding progressive growing blocks and RGB layers
        for i in range(
            len(factors) - 1  # Adjust for index error due to factors[i + 1]
        ):
            # Calculate input and output channels for the ConvBlock
            conv_in_c = int(in_channels * factors[i])
            conv_out_c = int(in_channels * factors[i + 1])

            # Add ConvBlock and RGB layer
            self.prog_blocks.append(ConvBlock(conv_in_c, conv_out_c))
            self.rgb_layers.append(
                WSConv2d(conv_out_c, img_channels, kernel_size=1, stride=1, padding=0)
            )

    def fade_in(self, alpha, upscaled, generated):
        """
        Implements the fade-in mechanism for blending images during progressive growing.
        
        Parameters:
        - alpha: Scalar within [0, 1] for blending control.
        - upscaled: Image from the previous resolution (upscaled).
        - generated: Image from the current resolution (generated).

        Returns:
        - Blended image.
        """
        return torch.tanh(alpha * generated + (1 - alpha) * upscaled)

    def forward(self, x, class_idx, alpha, steps):
        """
        Forward pass of the generator.
        
        Parameters:
        - x: Noise vector (batch_size, z_dim, 1, 1).
        - class_idx: Class indices for conditional generation.
        - alpha: Blend factor for fade-in mechanism.
        - steps: Number of steps for progressive growing (resolution stages).

        Returns:
        - Generated image.
        """
        class_idx = class_idx.long()

        # Embed the class indices and expand to match feature dimensions
        class_emb = self.class_embedding(class_idx)
        class_emb = class_emb.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, 1, 1).to(device)

        # Concatenate noise vector with class embedding
        x = torch.cat([x, class_emb], dim=1)

        # Pass through the initial block
        out = self.initial(x)

        # Handle the base case (no progressive growing)
        if steps == 0:
            return self.initial_rgb(out)

        # Progressive growing: Iterate through blocks based on steps
        for step in range(steps):
            # Upscale the feature map
            upscaled = F.interpolate(out, scale_factor=2, mode="nearest")

            # Apply the progressive block
            out = self.prog_blocks[step](upscaled, step)

        # Get RGB outputs from the last two resolutions
        final_upscaled = self.rgb_layers[steps - 1](upscaled)
        final_out = self.rgb_layers[steps](out)

        # Blend the two resolutions using fade-in
        return self.fade_in(alpha, final_upscaled, final_out)