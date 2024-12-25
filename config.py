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
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size for training")
    parser.add_argument("--learning_rate", type=float, default=0.0002, help="Learning rate for the optimizer")
    parser.add_argument("--epochs", type=int, default=100, help="Number of training epochs")
    parser.add_argument("--grad_penalty", type=float, default=10.0, help="Gradient penalty for training")
    parser.add_argument("--in_channel", type=int, default=512, help="Number of input channels")
    parser.add_argument("--save_model_path", type=str, default="./model.pth", help="Path to save the trained model")

    return parser.parse_args()
