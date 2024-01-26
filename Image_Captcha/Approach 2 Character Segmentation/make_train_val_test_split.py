# -*- coding: utf-8 -*-
"""
Created on Fri Oct 13 04:48:42 2023

"""

import os
import glob
import shutil
import sklearn.model_selection as skms
import concurrent.futures
from tqdm import tqdm

base_path = r'/users/pgrad/singhr6'
cwd = r'/users/pgrad/singhr6/segment_data2' # Make chanegs to point to directory where you have stored character segmented output data of preprocess_dataset.py
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

def encode_special_characters(filename):
    for char, mask in char_to_mask.items():
        filename = filename.replace(char, mask)
    return filename

def decode_special_characters(encoded_filename):
    for char, mask in char_to_mask.items():
        encoded_filename = encoded_filename.replace(mask, char)
    return encoded_filename

def copy_image(path, split_identifier):
    img_fn = os.path.split(path)[1] # e.g. "A_1583.png"
    img_label = img_fn.split(".png")[0]
    img_label = img_label.rsplit("_", 1)[0] # e.g. A, B, C, ..., 1, 2, 3, ...
    output_path = os.path.join(cwd, "data", "characters", split_identifier, img_label, img_fn)
    print("Output Path : ")
    print(output_path)
    shutil.copyfile(path, output_path)

def move_images_to_split(paths, split_identifier, char_set): # split = train, test, val
    # clear previous data
    if os.path.exists(os.path.join(cwd, "data", "characters", split_identifier)):
        shutil.rmtree(os.path.join(cwd, "data", "characters", split_identifier))

    # create data/characters/train, data/characters/test, and data/characters/val
    split_path = os.path.join(cwd, "data", "characters", split_identifier)
    if not os.path.exists(split_path):
        os.mkdir(split_path)

    # e.g. create data/characters/train/A, data/characters/train/B, etc...
    for char in char_set:
        char = encode_special_characters(char)
        split_char_path = os.path.join(cwd, "data", "characters", split_identifier, char)
        if not os.path.exists(split_char_path):
            os.mkdir(split_char_path)

    # move all character images into their respective split directory
    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = []
        for index, path in enumerate(paths):
            if (index+1) % 1000 == 0:
                print("{}: {} / {} ({}%)".format(split_identifier, index+1, len(paths), round((index+1) / len(paths) * 100.0, 2)))
            future = executor.submit(copy_image, path, split_identifier)
            futures.append(future)
        
        # Wait for all futures to complete
        if split_identifier == "val":
            concurrent.futures.wait(futures)

# find all characters used in the classification task (26 letters + 10 digits = 36 total)
uppercase_letters = [chr(ascii_val) for ascii_val in range(ord('A'), ord('Z') + 1)]

# Lowercase Letters
lowercase_letters = [chr(ascii_val) for ascii_val in range(ord('a'), ord('z') + 1)]

# Numbers
numbers = [chr(ascii_val) for ascii_val in range(ord('0'), ord('9') + 1)]

# Special Characters
special_characters = "-=_+[]{}\\|;':\",.<>/?`~!@#$%^&*()"

# Concatenate all character sets
char_set = uppercase_letters + lowercase_letters + numbers + list(special_characters)

# find paths to all extracted digits
characters_search_string = os.path.join(cwd, "data", "characters", "all_chars", "**", "*.png")
character_paths = glob.glob(characters_search_string, recursive = True)

# 90% train / 10% validation
# We split 10% validation and 90% for train
TRAIN_SIZE = int(0.9 * len(character_paths)) # 90% of data is used for train and validation
VAL_SIZE = int(0.1 * len(character_paths)) # Validation set is taken from the training set

train_paths, val_paths = skms.train_test_split(character_paths, train_size = TRAIN_SIZE)

move_images_to_split(train_paths, "train", char_set)
move_images_to_split(val_paths, "val", char_set)
