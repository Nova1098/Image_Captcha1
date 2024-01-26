# -*- coding: utf-8 -*-
"""
Created on Fri Oct 13 03:28:50 2023

@author: rahul
"""

import os
import cv2
import glob
import shutil
import scipy.ndimage
import numpy as np
import concurrent.futures
from noise_remover import NoiseRemover
from character_segmenter import CharacterSegmenter

base_path = r'/users/pgrad/singhr6'
cwd = r'/users/pgrad/singhr6/segment_data2'  # Make changes to point to where you want to store the output of the segmented characters
os.chdir(cwd)

char_to_mask = {
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

#if os.path.exists(os.path.join(cwd, "data", "characters", "all_chars")):
#    shutil.rmtree(os.path.join(cwd, "data", "characters", "all_chars")) # clear previous data

char_counts = {char: 0 for char in char_set}

def encode_special_characters(filename):
    for char, mask in char_to_mask.items():
        filename = filename.replace(char, mask)
    return filename

def decode_special_characters(encoded_filename):
    for char, mask in char_to_mask.items():
        encoded_filename = encoded_filename.replace(mask, char)
    return encoded_filename

def process_captcha(captcha_path):
    try:
        # image meta-details
        img_fn = os.path.split(captcha_path)[1]
        captcha_label = img_fn.split(".png")[0]
        captcha_label = decode_special_characters(captcha_label)
        print("Original CAPTCHA Label:", captcha_label)

        img = cv2.imread(captcha_path, cv2.IMREAD_GRAYSCALE)
        _, img = cv2.threshold(img, 230, 255, cv2.THRESH_BINARY)

        clean_image = NoiseRemover.remove_all_noise(img)

        masks, mask_sizes, mask_start_indices, mask_char_pixels_arrs = CharacterSegmenter.get_components(clean_image)
        if len(masks) == 0:
            return

        masks, mask_start_indices = CharacterSegmenter.segment_characters(masks, mask_sizes, mask_start_indices,
                                                                           mask_char_pixels_arrs)
        if not len(masks) > 0:
            return

        mask_start_indices, indices = zip(
            *sorted(zip(mask_start_indices, [i for i in range(len(mask_start_indices))])))
        masks = [masks[i] for i in indices]

        min_pixels_threshold = 20
        valid_masks = []
        valid_labels = []
        for mask, label in zip(masks, captcha_label):
            if np.sum(mask == 0) > min_pixels_threshold:
                valid_masks.append(mask)
                valid_labels.append(label)

        split_masks = []
        split_labels = []
        for mask, label in zip(valid_masks, valid_labels):
            intervals, _ = CharacterSegmenter.find_nonzero_intervals(np.sum(mask == 0, axis=0))
            if len(intervals) > 1:
                for interval in intervals:
                    start, end = interval
                    char_mask = mask[:, start:end]
                    split_masks.append(char_mask)
                    split_labels.append(label)
            else:
                split_masks.append(mask)
                split_labels.append(label)

        char_infos = [(split_masks[i], captcha_label[i]) for i in range(len(split_masks))]

        for index, char_info in enumerate(char_infos):
            char_crop, label = char_info

            char_crop = CharacterSegmenter.squarify_image(char_crop)
            char_crop = ~char_crop
            
            char_save_path = os.path.join(cwd, "data", "characters", "all_chars", encode_special_characters(label), "{}_{}.png".format(encode_special_characters(label), str(char_counts[label])))
            char_counts[label] +=1
            cv2.imwrite(char_save_path, char_crop)

        print("Processed {}/{} ({}%) CAPTCHAs...".format(captcha_paths.index(captcha_path) + 1, len(captcha_paths),
                                                         round((captcha_paths.index(captcha_path) + 1) / len(
                                                             captcha_paths) * 100.0, 2)))
    except Exception as e:
        print(f"Error processing captcha {captcha_path}: {e}")


# check if 'characters' folder exists, as well as folders for each digit individually
characters_dataset_path = os.path.join(cwd, "data", "characters")
if not os.path.exists(characters_dataset_path):
    os.mkdir(characters_dataset_path)
characters_dataset_split_path = os.path.join(cwd, "data", "characters", "all_chars")
if not os.path.exists(characters_dataset_split_path):
    os.mkdir(characters_dataset_split_path)
for char in char_set:
    char = encode_special_characters(char)
    char_folder_path = os.path.join(cwd, "data", "characters", "all_chars", char)
    if not os.path.exists(char_folder_path):
        os.mkdir(char_folder_path)

# Set the maximum number of concurrent threads based on your system's capabilities
MAX_THREADS = 16  # Adjust this number based on your system

# Use ThreadPoolExecutor for concurrent processing with a maximum of MAX_THREADS threads
with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_THREADS) as executor:
    captchas_path = os.path.join(base_path, "Desktop", "ValidateImage", "*.png") # path to all CAPTCHAs        # Make changes to point to train dataset and validete dataset with images to do the segmentation
    captcha_paths = glob.glob(captchas_path) # path to individual CAPTCHAs

    # Submit tasks to the executor for each captcha image
    futures = [executor.submit(process_captcha, captcha_path) for captcha_path in captcha_paths]

    # Wait for all tasks to complete
    concurrent.futures.wait(futures)

print("All CAPTCHAs processed.")
    
