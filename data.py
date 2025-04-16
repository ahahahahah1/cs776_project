# prompt: make all images of equal size of 512x512 for uniformity

from PIL import Image
import os

def resize_images(directory, target_size=(256, 256)):
    """Resizes all images in a directory to a target size.

    Args:
        directory: The path to the directory containing the images.
        target_size: A tuple (width, height) specifying the target size.
    """
    for filename in os.listdir(directory):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
            filepath = os.path.join(directory, filename)
            try:
                img = Image.open(filepath)
                img = img.resize(target_size, Image.LANCZOS)  # Use high-quality resampling
                img.save(filepath)  # Overwrite the original file
                print(f"Resized image: {filename}")
            except IOError:
                print(f"Error opening image file: {filepath}")

# Example usage:
# dataset_path = "./DIV2K_train_HR/DIV2K_train_HR/"  # Replace with your dataset path
dataset_path = "./DIV2K_valid_HR/DIV2K_valid_HR/"  # Replace with your dataset path
resize_images(dataset_path)
