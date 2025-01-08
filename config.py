# config.py
import argparse

# agrguments for inference
def get_inference_config():
    parser = argparse.ArgumentParser(description="Configuration for inference")
    # Inference-specific configurations
    parser.add_argument("--step", type=int, default=4, help="Step for resolution (default: 4 for 64x64)")
    parser.add_argument("--num_classes", type=int, default=35, help="Number of classes in the dataset")
    return parser.parse_args()

def get_training_config():
    parser = argparse.ArgumentParser(description="Configuration for training")

    # Training-specific configurations
    parser.add_argument("--progressive_epochs", type = list, default = [50, 50, 45, 45, 35, 30, 30, 25], help="Progressive epochs")
    parser.add_argument("--num_classes", type = int, default = 35, help="Total classes in dataset")
    parser.add_argument("--batch_sizes", type = list, default = [16, 16, 16, 16, 10, 10, 8, 8], help = "Batch sizes")
    parser.add_argument("--image_size", type = int, default = 128, help="Image size" )
    parser.add_argument("--lambda_gp", type=float, default=10.0, help="Gradient penalty for training")
    parser.add_argument("--start_train_at_img_size", type=int, default=8, help="Start training with img size")
    parser.add_argument("--learning_rate", type = float, default=1e-3, help="Learning Rate")


    return parser.parse_args()
