# import requirements
import re
import torch                                                          # type:ignore
from torch import nn                                                  # type:ignore
import numpy as np                                                    # type:ignore
from attention import Self_Attention
from config import get_inference_config
import torch.nn.functional as F                                       # type:ignore
import torchvision.utils as vutils                                    # type:ignore
from PIL import Image, ImageDraw, ImageFont


# Initialize important variables
img_channels = 3
in_channel = 512 
z_dim = 512
grad_penality = 10
factors = [1, 1, 1, 1, 1 / 2, 1 / 4, 1 / 8, 1 / 16, 1 / 32]
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load another variables from config 
args = get_inference_config()
step = args.step
num_classes = args.num_classes

# Make a class dictionary of all letters.
classes_dict = {'1': 1, '2': 2, '3': 3, '4': 4, '5': 5, '6': 6, '7': 7, '8': 8, '9': 9, 'A': 10, 
                'B': 11, 'C': 12, 'D': 13, 'E': 14, 'F': 15, 'G': 16, 'H': 17, 'I': 18, 'J': 19, 'K': 20,
                'L': 21, 'M': 22, 'N': 23, 'O': 24, 'P': 25, 'Q': 26, 'R': 27, 'S': 28, 'T': 29, 'U': 30,
                'V': 31, 'W': 32, 'X': 33, 'Y': 34, 'Z': 35}


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

# Define the Generator class
class Generator(nn.Module):
    def __init__(self, z_dim, in_channels, num_classes, img_channels=3):
        """
        Initializes the Generator module.

        Args:
            z_dim (int): Dimension of the input latent vector (noise).
            in_channels (int): Number of base input channels.
            num_classes (int): Number of classes for conditional generation.
            img_channels (int): Number of output image channels (e.g., 3 for RGB). Default is 3.
        """
        super(Generator, self).__init__()
        # Embedding layer to represent class labels as feature vectors
        self.class_embedding = nn.Embedding(num_classes, num_classes)
        # Initial block for generating feature maps from latent vector and class embedding
        self.initial = nn.Sequential(
            PixelNorm(),  # Normalize the latent vector for stable training
            nn.ConvTranspose2d(z_dim + num_classes, in_channels, 4, 1, 0),  # Upsample latent vector
            nn.LeakyReLU(0.2),  # Activation function
            WSConv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1),  # Weight-standardized convolution
            nn.LeakyReLU(0.2),  # Activation function
            PixelNorm(),  # Normalize feature maps
        )
        # Initial layer for converting feature maps to RGB images
        self.initial_rgb = WSConv2d(
            in_channels, img_channels, kernel_size=1, stride=1, padding=0
        )
        # Progressive blocks and RGB layers for progressively growing the generator
        self.prog_blocks, self.rgb_layers = (
            nn.ModuleList([]),  # Blocks for generating higher-resolution features
            nn.ModuleList([self.initial_rgb]),  # Layers for converting features to RGB
        )
        # Define progressive convolutional blocks
        for i in range(len(factors) - 1):  # Avoid index error with factors[i+1]
            conv_in_c = int(in_channels * factors[i])  # Input channels for the current block
            conv_out_c = int(in_channels * factors[i + 1])  # Output channels for the next block
            self.prog_blocks.append(ConvBlock(conv_in_c, conv_out_c))  # Add ConvBlock
            self.rgb_layers.append(
                WSConv2d(conv_out_c, img_channels, kernel_size=1, stride=1, padding=0)
            )  # Add RGB layer
    def fade_in(self, alpha, upscaled, generated):
        """
        Performs a fade-in operation during progressive training.
        Args:
            alpha (float): Transition factor between two resolutions (0 to 1).
            upscaled (tensor): Upscaled image from the previous resolution.
            generated (tensor): Generated image at the current resolution.
        Returns:
            tensor: Blended image with a smooth transition.
        """
        return torch.tanh(alpha * generated + (1 - alpha) * upscaled)

    def forward(self, x, class_idx, alpha, steps):
        """
        Forward pass for the generator.
        Args:
            x (tensor): Latent vector (noise) of shape (B, z_dim, 1, 1).
            class_idx (tensor): Class indices for conditional generation (B,).
            alpha (float): Transition factor for progressive fade-in.
            steps (int): Number of progressive steps (0-based).

        Returns:
            tensor: Generated image of the specified resolution.
        """
        # Convert class indices to embeddings and reshape for concatenation
        class_idx = class_idx.long()  # Ensure indices are integers
        class_emb = self.class_embedding(class_idx)  # Generate class embeddings
        class_emb = class_emb.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, 1, 1).to(device)
        # Concatenate latent vector with class embedding
        x = torch.cat([x, class_emb], dim=1)
        # Pass through the initial block
        out = self.initial(x)
        # If at the first step, directly generate the initial RGB image
        if steps == 0:
            return self.initial_rgb(out)
        # Iterate through progressive blocks and upscale feature maps
        for step in range(steps):
            upscaled = F.interpolate(out, scale_factor=2, mode="nearest")  # Upscale feature maps
            out = self.prog_blocks[step](upscaled, step)  # Apply ConvBlock for the current step
        # Generate RGB images from the upscaled and current outputs
        final_upscaled = self.rgb_layers[steps - 1](upscaled)  # Upscaled image
        final_out = self.rgb_layers[steps](out)  # Current output image
        # Perform fade-in to smoothly transition between resolutions
        return self.fade_in(alpha, final_upscaled, final_out)
    

## Boilerplate code
epoch = 19
gen = Generator(
    z_dim, in_channel, num_classes, img_channels=img_channels
).to(device)
gen_checkpoint_path = f'/media/ajeet/B4363DE3363DA770/Users/Ajeet/Downloads/Masters/Main-Project/Publication/git/Ajeet/all_experiments/checkpoint_G_{4 * 2 ** step}_{epoch}.pth'
checkpoint_G = torch.load(gen_checkpoint_path)
gen.load_state_dict(checkpoint_G['model_state_dict'])
alpha = checkpoint_G['alpha']

# Take the input from user
def takeInput():
    """
    Takes user input for generating signs and processes the input.
    Returns:
        cleaned_words (list): A list of cleaned and uppercase words 
        split from the user input, removing any unwanted characters.
    """
    in_str = input("Enter what you want to generate in signs: ")  # Prompt user input
    val_str = in_str.upper().split(" ")  # Convert to uppercase and split into words
    cleaned_words = []  # Initialize a list to store cleaned words
    # Process each word to remove unwanted characters
    for x in val_str:
        cleaned_word = re.sub(r'[^a-zA-Z1-9]', '', x)  # Remove non-alphanumeric characters
        if cleaned_word:  # Add to the list if the word is not empty
            cleaned_words.append(cleaned_word)
    return cleaned_words  # Return the cleaned and processed words

def image_With_Labels(sample, labels, save_path, nrow=8, black_image_size=64):
    """
    Creates an image grid with corresponding labels overlaid on each image.
    Args:
        sample (Tensor): A batch of images (N, C, H, W) to arrange in a grid.
        labels (list): List of labels corresponding to the images.
        save_path (str): Path to save the labeled image grid.
        nrow (int): Number of images per row in the grid. Default is 8.
        black_image_size (int): Size of a single image in the grid (not directly used). Default is 64.
    Returns:
        None: Saves the labeled image grid to the specified path.
    """
    # Convert integer labels to their string representation
    labels = [str(label).lower() for label in labels]  # Convert labels to lowercase strings
    num_image = sample.size(0)  # Get the number of images in the sample batch
    image_size = sample.size(2)  # Assuming square images, retrieve the height/width
    # Create an image grid using torchvision's utility function
    image_grid = vutils.make_grid(sample, nrow=nrow, normalize=True, padding=2)
    # Convert the image grid to a PIL image for drawing labels
    image_grid_pil = Image.fromarray((image_grid.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8))
    # Initialize the font and drawing context
    custom_font = ImageFont.truetype(
        '/media/ajeet/B4363DE3363DA770/Users/Ajeet/Downloads/Masters/Main-Project/Publication/git/Ajeet/Generation-of-Indian-Sign-Language-Letters-and-Numbers-main/AurulentSansMono-Regular.otf',
        size=14
    )
    draw = ImageDraw.Draw(image_grid_pil)
    # Loop through all images and place their corresponding labels
    for i in range(num_image):
        row = i // nrow  # Determine the row in the grid
        col = i % nrow   # Determine the column in the grid
        # Calculate x and y positions for the label
        x = col * (image_size + 2) + 3  # Horizontal position
        y = row * (image_size + 2) + 2  # Vertical position
        label = labels[i]  # Retrieve the label for the current image
        # Draw the label on the image grid
        draw.text((x, y), label, fill=(255, 255, 255), font=custom_font)
    # Save the labeled image grid to the specified path
    image_grid_pil.save(save_path)

# Example usage of the functions
with torch.no_grad():  # Disable gradient computation for efficiency
    # Get user input and clean it into words
    cleaned_words = takeInput()  # Returns a list of cleaned and processed words
    all_images = []  # List to store all generated images
    all_labels = []  # List to store labels corresponding to the images

    # Process each cleaned word
    for word in cleaned_words:
        num_image = len(word)  # Number of characters in the word
        # Convert each character to its corresponding class index (0-indexed)
        lst = [classes_dict[char] - 1 for char in word]
        labels = lst  # Store labels for the current word
        # Generate random noise vector for the generator
        noise = torch.randn(num_image, z_dim, 1, 1).to(device)  # Noise tensor for each character
        fake_labels = torch.tensor(labels, device=device)  # Convert labels to a tensor

        # Generate images using the generator model
        sample = gen(noise, fake_labels, alpha, step)

        # Append the generated images and labels to the lists
        all_images.append(sample)  # Add generated images to the list
        all_labels.extend(list(word))  # Add corresponding characters as labels

        # Add a blank image as a separator between words
        blank_image = torch.zeros(1, 3, sample.size(2), sample.size(2)).to(device)  # Create blank image
        all_images.append(blank_image - 0.7)  # Adjust the blank image brightness
        all_labels.append(' ')  # Add a blank label for the separator

    # Combine all images into a single tensor and remove the last blank image
    all_images = torch.cat(all_images[:-1], dim=0)  #
    save_path = f"./generated_image_{4 * 2 ** step}_2.png"  

    # Save the labeled image grid as a PDF
    image_With_Labels(all_images, all_labels, save_path, nrow=8)

    print("Image saved successfully!")  
