# -*- coding: utf-8 -*-
"""
Created on Sat Oct 21 00:46:27 2023

"""

###########################################################
## ReadME                  ################################
## Make changes wherever a comment is mentioned ###########
###########################################################

from noise_remover import NoiseRemover
import numpy as np
import glob
import cv2
import os
import concurrent.futures

# Define the base directory
base_directory = r'/users/pgrad/singhr6/'

# Change the current working directory
os.chdir(base_directory)

# Dictionary mapping special characters to their corresponding masks
CHARACTER_MASK_MAPPING = {
    '<': '_101_',
    '>': '_102_',
    ':': '_103_',
    '"': '_104_',
    '/': '_105_',
    '\\': '_106_',
    '|': '_107_',
    '?': '_108_',
    '*': '_109_',
    '.': '_111_'
}

def encode_special_characters(filename):
    # Replace special characters with corresponding masks
    for char, mask in CHARACTER_MASK_MAPPING.items():
        filename = filename.replace(char, mask)
    return filename

def decode_special_characters(encoded_filename):
    # Replace masks with original special characters
    for char, mask in CHARACTER_MASK_MAPPING.items():
        encoded_filename = encoded_filename.replace(mask, char)
    return encoded_filename

# Function to process and save images
def process_and_save_image(captcha_path):
    img_fn = os.path.split(captcha_path)[1]
    captcha_label = img_fn.split(".png")[0]
    captcha_label = decode_special_characters(captcha_label)
    print("Original CAPTCHA Label:", captcha_label)
    
    # Define output path for processed images
    processed_image_path = os.path.join(base_directory, "Desktop", "ProcessedTrainImage2", encode_special_characters(captcha_label) + ".png")
    # Make changes for a directory where you want to put the Processed Images. 
    
    img = cv2.imread(captcha_path, cv2.IMREAD_GRAYSCALE)
    _, img = cv2.threshold(img, 230, 255, cv2.THRESH_BINARY)

    cleaned_image = NoiseRemover.remove_all_noise(img)
    brightened_image = cv2.equalizeHist(cleaned_image)
    
    cv2.imwrite(processed_image_path, brightened_image)
    print(f"Processed image saved at: {processed_image_path}")

# Define the character sets
# Uppercase Letters
uppercase_letters = [chr(ascii_val) for ascii_val in range(ord('A'), ord('Z') + 1)]

# Lowercase Letters
lowercase_letters = [chr(ascii_val) for ascii_val in range(ord('a'), ord('z') + 1)]

# Numbers
numbers = [chr(ascii_val) for ascii_val in range(ord('0'), ord('9') + 1)]

# Special Characters
special_characters = "-=_+[]{}\\|;':\",.<>/?`~!@#$%^&*()"

# Concatenate all character sets
char_set = uppercase_letters + lowercase_letters + numbers + list(special_characters)

# Initialize char_counts dictionary
char_counts = {char: 0 for char in char_set}

# Check if the output directory exists, if not, create it
output_directory = os.path.join(base_directory, "Desktop", "ProcessedTrainImage2") # Make changes to point to directory where you want to store processed images
if not os.path.exists(output_directory):
    os.mkdir(output_directory)

# Path to CAPTCHA images
captchas_path = os.path.join(base_directory, "Desktop", "TrainImage2", "*.png")  # Make changes to the path where the current images are available
captcha_paths = glob.glob(captchas_path)  

# Use ThreadPoolExecutor to process and save images concurrently
with concurrent.futures.ThreadPoolExecutor() as executor:
    # Submit the tasks to the executor
    futures = [executor.submit(process_and_save_image, captcha_path) for captcha_path in captcha_paths]
    
    # Wait for all tasks to complete
    concurrent.futures.wait(futures)

print("All images processed and saved.")
