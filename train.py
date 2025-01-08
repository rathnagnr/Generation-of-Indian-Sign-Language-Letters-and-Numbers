# import requirements
import gc
import os
import torch                                                #type:ignore
from math import log2
from tqdm import tqdm                                       #type:ignore
from torch import optim                                     #type:ignore
import matplotlib.pyplot as plt                             #type:ignore
from generator import Generator
from critic import Discriminator
from torchvision.utils import save_image                    #type:ignore
from torchvision import datasets, transforms                #type:ignore
from torch.utils.data import DataLoader                     #type:ignore
from config import get_training_config



# Initialize important variables
# These control the progression in the generator/discriminator network
factors = [1, 1, 1, 1, 1 / 2, 1 / 4, 1 / 8, 1 / 16, 1 / 32]
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.cuda.init() 
# Number of image channels (e.g., 3 for RGB images)
channels_img = 3
# Latent vector size (Z-dimension) used in the generator
z_dim = 512 
# Number of input channels for the network
in_channels = 512  


# Load another variables from config 
args = get_training_config()
progressive_epochs = args.progressive_epochs
num_classes = args.num_classes
batch_sizes = args.batch_sizes
image_size = args.image_size
lambda_gp = args.lambda_gp
start_train_at_img_size = args.start_train_at_img_size
learning_rate = args.learning_rate



# Function to create a data loader for a specific image size
def get_loader(image_size):
    """
    Loads the dataset and returns a DataLoader for the specified image size.
    Args:
        image_size (int): The target size for image resizing.
    Returns:
        DataLoader: A PyTorch DataLoader for the dataset.
        Dataset: The underlying dataset.
    """
    # Define the transformations for preprocessing the dataset
    transform = transforms.Compose(
        [
            # Resize the image to the target resolution
            transforms.Resize((image_size, image_size)),
            # Convert image to PyTorch tensor
            transforms.ToTensor(),
            # Normalize the image to range [-1, 1] for all channels
            transforms.Normalize(
                [0.5 for _ in range(3)],  # Mean
                [0.5 for _ in range(3)],  # Standard deviation
            )
        ]
    )
    
    # Determine the batch size based on the image resolution
    batch_size = batch_sizes[int(log2(image_size / 4))]  # Uses log2 for indexing
    # Path to the dataset folder
    path = '/home2/nishantk/Ajeet/pro_attention/data'
    # Load dataset using ImageFolder, applying the defined transformations
    dataset = datasets.ImageFolder(root=path, transform=transform)
    # Create DataLoader for batching and shuffling the data
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=18  # Number of parallel workers for data loading
    )
    return loader, dataset

# Function to visualize some real samples from the dataset
def check_loader(image_size):
    """
    Visualizes a batch of images from the DataLoader.
    Args:
        image_size (int): The image size for the DataLoader.
    """
    # Get a DataLoader for the specified image size
    loader, _ = get_loader(image_size)
    # Fetch one batch of images
    cloth, _ = next(iter(loader))
    # Create a 3x3 grid for displaying images
    _, ax = plt.subplots(3, 3, figsize=(8, 8))
    # Title for the visualization
    plt.suptitle('Some real samples', fontsize=15, fontweight='bold')
    ind = -2  # Start index for images
    for k in range(3):  # Rows
        for kk in range(3):  # Columns
            ind += 1
            # Display the image after scaling pixel values back to [0, 1]
            ax[k][kk].imshow((cloth[ind].permute(1, 2, 0) + 1) / 2)

check_loader(image_size)

# Gradient penality function
def gradient_penalty(critic, real, fake, real_labels, alpha, train_step, device="cpu"):
    """
    Computes the gradient penalty for Wasserstein GAN with Gradient Penalty (WGAN-GP).
    Args:
        critic: The discriminator/critic model.
        real: Real images from the dataset.
        fake: Generated images from the generator.
        real_labels: Labels for real images.
    Returns:
        Gradient penalty term.
    """
    batch_size, C, H, W = real.shape
    # Interpolate between real and fake images
    beta = torch.rand((batch_size, 1, 1, 1)).repeat(1, C, H, W).to(device)
    interpolated_images = real * beta + fake.detach() * (1 - beta)
    interpolated_images.requires_grad_(True)

    # Compute critic scores for interpolated images
    mixed_scores = critic(interpolated_images, real_labels, alpha, train_step)
    # Compute gradients of scores with respect to interpolated images
    gradient = torch.autograd.grad(
        inputs=interpolated_images,
        outputs=mixed_scores,
        grad_outputs=torch.ones_like(mixed_scores),
        create_graph=True,
        retain_graph=True,
    )[0]
    # Calculate the gradient norm and penalty
    gradient = gradient.view(gradient.shape[0], -1)
    gradient_norm = gradient.norm(2, dim=1)
    gradient_penalty = torch.mean((gradient_norm - 1) ** 2)
    return gradient_penalty


def generate_examples(gen, steps, epoch, n):
    """
    Generates and saves example images using the generator.
    Args:
        gen: The generator model.
        steps: Current training step for progressive GAN.
        epoch: Current training epoch.
        n: Number of images to generate.
    """
    gen.eval()  # Set generator to evaluation mode
    alpha = 1.0  # Full blending
    class_idx = torch.Tensor([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]).to(device)

    for i in range(n):
        with torch.no_grad():  # No gradient computation for inference
            noise = torch.randn(n, z_dim, 1, 1).to(device)  # Random noise for generation
            img = gen(noise, class_idx, alpha, steps)
            # Create output directory if it doesn't exist
            if not os.path.exists(f'./expl_attn/step{steps}'):
                os.makedirs(f'./expl_attn/step{steps}')
            # Save generated images scaled back to [0, 1]
            save_image(img * 0.5 + 0.5, f"./expl_attn/step{steps}/{epoch}_{i}.png")
    gen.train()  # Set generator back to training mode


def train_fn(
    critic,
    gen,
    loader,
    dataset,
    step,
    alpha,
    opt_critic,
    opt_gen,
):
    loop = tqdm(loader, leave=True)
    for batch_idx, (real, real_labels) in enumerate(loop):
        real, real_labels = real.to(device), real_labels.to(device)
        cur_batch_size = real.shape[0]
        # Train Critic
        noise = torch.randn(cur_batch_size, z_dim, 1, 1).to(device)
        noise_labels = torch.randint(0, num_classes, (cur_batch_size,), device=device)
        fake = gen(noise, noise_labels, alpha, step)
        critic_real = critic(real, real_labels, alpha, step)
        critic_fake = critic(fake.detach(), noise_labels, alpha, step)
        gp = gradient_penalty(critic, real, fake, real_labels, alpha, step, device=device)
        # Compute critic loss
        loss_critic = (
            -(torch.mean(critic_real) - torch.mean(critic_fake))
            + lambda_gp * gp
            + (0.001 * torch.mean(critic_real ** 2))  # Regularization term
        )
        critic.zero_grad()
        loss_critic.backward()
        opt_critic.step()
        # Train Generator
        gen_fake = critic(fake, noise_labels, alpha, step)
        loss_gen = -torch.mean(gen_fake)
        gen.zero_grad()
        loss_gen.backward()
        opt_gen.step()
        # Update alpha for smooth transition in progressive training
        alpha += cur_batch_size / ((progressive_epochs[step] * 0.5) * len(dataset))
        alpha = min(alpha, 1)

        # Display training metrics
        loop.set_postfix(
            gp=gp.item(),
            loss_critic=loss_critic.item(),
        )
    return alpha


# Initialize Generator and Discriminator (critic as per WGAN terminology)
gen = Generator(z_dim, in_channels, num_classes, img_channels=channels_img).to(device)
critic = Discriminator(in_channels, num_classes, img_channels=channels_img).to(device)

# Initialize optimizers with Adam
opt_gen = optim.Adam(gen.parameters(), lr=learning_rate, betas=(0.0, 0.99))
opt_critic = optim.Adam(critic.parameters(), lr=learning_rate, betas=(0.0, 0.99))

# Set models to training mode
gen.train()
critic.train()


# Initialize step and epoch based on starting image size
step = 0
epoch = 0
# Check if checkpoints exist for resuming training. Make a directory by name ckpt_attn
gen_checkpoint_path = f'./ckpt_attn/checkpoint_G_{4 * 2 ** step}_{epoch}.pth'
critic_checkpoint_path = f'./ckpt_attn/checkpoint_D_{4 * 2 ** step}_{epoch}.pth'

if os.path.exists(gen_checkpoint_path) and os.path.exists(critic_checkpoint_path):
    # Load generator and critic checkpoints
    checkpoint_G = torch.load(gen_checkpoint_path)
    gen.load_state_dict(checkpoint_G['model_state_dict'])
    opt_gen.load_state_dict(checkpoint_G['optimizer_state_dict'])
    start_epoch_gen = checkpoint_G['epoch']

    checkpoint_D = torch.load(critic_checkpoint_path)
    critic.load_state_dict(checkpoint_D['model_state_dict'])
    opt_critic.load_state_dict(checkpoint_D['optimizer_state_dict'])
    start_epoch_critic = checkpoint_D['epoch']

    # Start from the furthest epoch saved
    start_epoch = max(start_epoch_gen, start_epoch_critic) + 1
    print(f"Resuming training from epoch {start_epoch}")
else:
    start_epoch = 0
    print("Starting training from scratch")

# Train across progressive image sizes
for k in range(len(progressive_epochs[step - 1:])):
    num_epochs = progressive_epochs[step - 1]
    alpha = 1e-5  # Initialize alpha for progressive training
    loader, dataset = get_loader(4 * 2 ** step)  # Get loader for current image size
    print(f"Current image size: {4 * 2 ** step}")
    # Train for remaining epochs
    for epoch in range(start_epoch, num_epochs):
        print(f"Epoch [{epoch + 1}/{num_epochs}]")
        alpha = train_fn(critic, gen, loader, dataset, step, alpha, opt_critic, opt_gen)
        # Save checkpoints at each epoch
        checkpoint_G = {
            'epoch': epoch,
            'alpha': alpha,
            'model_state_dict': gen.state_dict(),
            'optimizer_state_dict': opt_gen.state_dict()
        }
        torch.save(checkpoint_G, f'./ckpt_attn/checkpoint_G_{4 * 2 ** step}_{epoch}.pth')
        checkpoint_D = {
            'epoch': epoch,
            'model_state_dict': critic.state_dict(),
            'optimizer_state_dict': opt_critic.state_dict()
        }
        torch.save(checkpoint_D, f'./ckpt_attn/checkpoint_D_{4 * 2 ** step}_{epoch}.pth')
        # Generate examples for visualization
        generate_examples(gen, step, epoch, n=15)
    # Progress to the next image size
    step += 1
    start_epoch = 0  # Reset start epoch for the next size
    # Clear memory and cache
    del loader, dataset, alpha
    torch.cuda.empty_cache()
    gc.collect()

