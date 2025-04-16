import os
import json
import numpy as np
from PIL import Image
from tqdm import tqdm
from skimage import io, img_as_float
from skimage.util import random_noise
from skimage.restoration import denoise_tv_chambolle

# Define input and output directories
input_dir = 'DIV2K_valid_HR/DIV2K_valid_HR/'  # Replace with your input folder path
output_dir = 'DIV2K_valid_HR/benchmark_ready2/'  # Replace with your desired output folder path
# input_dir = 'DIV2K_train_HR/DIV2K_train_HR/'  # Replace with your input folder path
# output_dir = 'DIV2K_train_HR/benchmark_ready/'  # Replace with your desired output folder path
os.makedirs(output_dir, exist_ok=True)

# Define noise levels
noise_levels = [0.1, 0.15, 0.20, 0.30, 0.4]

# Initialize a list to store mappings
mappings = []

# Get list of image files
image_files = [f for f in os.listdir(input_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff'))]

# Calculate total number of iterations
total_iterations = len(image_files) * len(noise_levels)

# Create a progress bar
with tqdm(total=total_iterations, desc="Processing Images") as pbar:
    # Process each image in the input directory
    for filename in image_files:
        # Read and normalize the image
        image_path = os.path.join(input_dir, filename)
        image = img_as_float(io.imread(image_path))

        for sigma in noise_levels:
            # Add Gaussian noise
            noisy_image = random_noise(image, var=sigma**2)

            # Apply TV denoising
            denoised_image = denoise_tv_chambolle(noisy_image, weight=0.1, channel_axis=-1)

            # Define output filenames
            base_name, ext = os.path.splitext(filename)
            noisy_filename = f"{base_name}_noisy_{sigma}{ext}"
            denoised_filename = f"{base_name}_denoised_{sigma}{ext}"

            # Clip and convert images to uint8
            noisy_image_uint8 = (np.clip(noisy_image, 0, 1) * 255).astype(np.uint8)
            denoised_image_uint8 = (np.clip(denoised_image, 0, 1) * 255).astype(np.uint8)

            # Save the noisy and denoised images using Pillow
            Image.fromarray(noisy_image_uint8).save(os.path.join(output_dir, noisy_filename))
            Image.fromarray(denoised_image_uint8).save(os.path.join(output_dir, denoised_filename))

            # Append the mapping to the list
            mappings.append({
                'original_image': filename,
                'noise_level': sigma,
                'noisy_image': noisy_filename,
                'denoised_image': denoised_filename
            })

            # Update the progress bar
            pbar.update(1)

# Save the mappings to a JSON file
with open(os.path.join(output_dir, 'image_mappings.json'), 'w') as json_file:
    json.dump(mappings, json_file, indent=4)

# https://drive.google.com/file/d/1G9ValzmzSRBOZlv2wKef2nxla2CQQNjk/view?usp=sharing