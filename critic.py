# import requirements
import torch                                         #type:ignore
from torch import nn                                 #type:ignore
from attention import Self_Attention
from generator import factors
from attention import ConvBlock, WSConv2d


class Discriminator(nn.Module):
    def __init__(self, in_channels, num_classes, img_channels=3):
        """
        Initializes the Discriminator model.
        
        Parameters:
        - in_channels: Number of channels in the intermediate feature maps.
        - num_classes: Number of distinct classes for conditional discrimination.
        - img_channels: Number of channels in the input image (default: 3 for RGB images).
        """
        super(Discriminator, self).__init__()
        # Progressive blocks and RGB layers to process input at different resolutions
        self.prog_blocks, self.rgb_layers = nn.ModuleList([]), nn.ModuleList([])
        # Activation function
        self.leaky = nn.LeakyReLU(0.2)
        # Embedding layer for class labels
        self.class_embedding = nn.Embedding(num_classes, num_classes)
        # Linear layers to map class embeddings to spatial features for different resolutions
        self.linear0 = nn.Linear(num_classes, 16)
        self.linear1 = nn.Linear(num_classes, 64)
        self.linear2 = nn.Linear(num_classes, 1 * 16 * 16)
        self.linear3 = nn.Linear(num_classes, 1 * 32 * 32)
        self.linear4 = nn.Linear(num_classes, 1 * 64 * 64)
        self.linear5 = nn.Linear(num_classes, 1 * 128 * 128)
        self.linear6 = nn.Linear(num_classes, 1 * 256 * 256)
        self.linear7 = nn.Linear(num_classes, 1 * 512 * 512)
        self.linear8 = nn.Linear(num_classes, 1 * 1024 * 1024)

        # Self-attention modules to enhance feature learning
        self.attention1 = Self_Attention(512, 'relu')
        self.attention2 = Self_Attention(512, 'relu')

        # Mirror the generator for progressive blocks and RGB layers (work backward from large to small resolution)
        for i in range(len(factors) - 1, 0, -1):
            conv_in = int(in_channels * factors[i])
            conv_out = int(in_channels * factors[i - 1])

            # Progressive block for downsampling and feature extraction
            self.prog_blocks.append(ConvBlock(conv_in, conv_out, use_pixelnorm=False))

            # RGB layer to process input images with an additional channel for class label
            self.rgb_layers.append(
                WSConv2d(img_channels + 1, conv_in, kernel_size=1, stride=1, padding=0)
            )

        # Initial RGB layer for 4x4 resolution input
        self.initial_rgb = WSConv2d(
            img_channels + 1, in_channels, kernel_size=1, stride=1, padding=0
        )
        self.rgb_layers.append(self.initial_rgb)

        # Average pooling layer for downsampling
        self.avg_pool = nn.AvgPool2d(kernel_size=2, stride=2)

        # Final block for 4x4 resolution input
        self.final_block = nn.Sequential(
            WSConv2d(in_channels + 1, in_channels, kernel_size=3, padding=1),  # Minibatch std added
            nn.LeakyReLU(0.2),
            WSConv2d(in_channels, in_channels, kernel_size=4, padding=0, stride=1),
            nn.LeakyReLU(0.2),
            WSConv2d(in_channels, 1, kernel_size=1, padding=0, stride=1),  # Produces final logits
        )

    def fade_in(self, alpha, downscaled, out):
        """
        Implements the fade-in mechanism for blending inputs during progressive growing.
        Parameters:
        - alpha: Blend factor
        - downscaled: Input from a lower resolution.
        - out: Output from the current resolution.
        Returns:
        - Blended result.
        """
        return alpha * out + (1 - alpha) * downscaled

    def minibatch_std(self, x):
        """
        Calculates minibatch standard deviation and appends it as an additional feature channel.
        Parameters:
        - x: Input tensor.
        Returns:
        - Tensor with minibatch standard deviation added as a feature channel.
        """
        batch_statistics = (
            torch.std(x, dim=0, unbiased=False).mean().repeat(x.shape[0], 1, x.shape[2], x.shape[3])
        )
        return torch.cat([x, batch_statistics], dim=1)

    def forward(self, x, labels, alpha, steps):
        """
        Forward pass of the discriminator.
        Parameters:
        - x: Input image tensor.
        - labels: Class labels for conditional discrimination.
        - steps: Number of steps for progressive growing (resolution stages).
        Returns:
        - Discriminator logits for the input images.
        """
        cur_step = len(self.prog_blocks) - steps

        # Embed the labels and map them to spatial features for the current resolution
        label_embed = self.class_embedding(labels)
        class_linear = getattr(self, f'linear{steps}')(label_embed)
        x = torch.cat([x, class_linear.view(x.size(0), 1, *x.shape[2:])], dim=1)

        # Initial processing with the corresponding RGB layer
        out = self.leaky(self.rgb_layers[cur_step](x))
        if steps == 0:
            # For the smallest resolution, add minibatch standard deviation and use the final block
            out = self.minibatch_std(out)
            return self.final_block(out).view(out.shape[0], -1)

        # Downscale and process through progressive blocks
        downscaled = self.leaky(self.rgb_layers[cur_step + 1](self.avg_pool(x)))
        out = self.avg_pool(self.prog_blocks[cur_step](out, step=0))
        out = self.fade_in(alpha, downscaled, out)

        # Process through subsequent blocks
        for step in range(cur_step + 1, len(self.prog_blocks)):
            out = self.prog_blocks[step](out, step=0)
            if step == 4:
                out = self.attention1(out)  # Apply self-attention at step 4
            # elif step ==5:
            #     out = self.attention2
            out = self.avg_pool(out)

        # Add minibatch standard deviation and process through the final block
        out = self.minibatch_std(out)
        return self.final_block(out).view(out.shape[0], -1)