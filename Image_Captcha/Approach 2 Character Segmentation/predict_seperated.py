# -*- coding: utf-8 -*-
"""
Created on Sat Oct 21 18:22:36 2023

"""
###########################################################
## ReadME                  ################################
## Make changes wherever a comment is mentioned ###########
###########################################################

import os
import cv2
import torch
import torch.nn as nn

from datasets import CharactersDataset, CAPTCHADataset
from model import CharacterClassifier

from noise_remover import NoiseRemover
from character_segmenter import CharacterSegmenter

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

def preprocess_image(image_path):
    print(image_path)
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    _, img = cv2.threshold(img, 230, 255, cv2.THRESH_BINARY)

    # clean up the image by removing noise
    clean_image = NoiseRemover.remove_all_noise(img)

    masks, mask_sizes, mask_start_indices, mask_char_pixels_arrs = CharacterSegmenter.get_components(clean_image)
    if len(masks) == 0:
        print("Bad Image found")
        return

    # segment and extract characters
    masks, mask_start_indices = CharacterSegmenter.segment_characters(masks, mask_sizes, mask_start_indices, mask_char_pixels_arrs)
    if not len(masks) > 0:
        print("Bad Image found")
        return

    # reorder masks and starting indices in ascending order to align them with the proper character for labeling
    mask_start_indices, indices = zip(*sorted(zip(mask_start_indices, [i for i in range(len(mask_start_indices))]))) # make sure intervals are in left-to-right order so we can segment characters properly
    masks = [masks[i] for i in indices]
    
    min_pixels_threshold = 20
    # Filter out masks that don't correspond to valid characters (for example, masks with too few pixels)
    valid_masks = []
    for mask, label in zip(masks):
        if np.sum(mask == 0) > min_pixels_threshold:  # Minimum number of black pixels to consider it a valid character
            valid_masks.append(mask)

    # Split masks that correspond to multiple characters
    split_masks = []
    for mask in zip(valid_masks):
        intervals, _ = CharacterSegmenter.find_nonzero_intervals(np.sum(mask == 0, axis=0))
        if len(intervals) > 1:  # If the mask contains multiple segments, split it
            for interval in intervals:
                start, end = interval
                char_mask = mask[:, start:end]
                split_masks.append(char_mask)
        else:
            split_masks.append(mask)

    char_infos = [(split_masks[i]) for i in range(len(split_masks))]

    return char_infos


def callthis(model_root_dir, model_name, captcha_dir, output_file):
    
    # initialize model and load trained weights
    model = CharacterClassifier(num_classes = 94, pretrained = False).to(device)
    model.load_state_dict(torch.load(os.path.join(model_root_dir, model_name)))
    model.to(device)

    list_dir =  os.listdir(captcha_dir)
    
    #locale.setlocale(locale.LC_COLLATE, 'de_DE.UTF-8')
    
    sorted_list_dir = sorted(list_dir, key=lambda x: [ord(c) for c in x])
    
    for x in sorted_list_dir:
        # load image and preprocess it
        processed_image_characters = preprocess_image(x)
        
        predicted_captcha = ""
        for index, char_info in enumerate(processed_image_characters):
            char_crop = char_info

            # reshape character crop to 76x76
            char_crop = CharacterSegmenter.squarify_image(char_crop)
            char_crop = ~char_crop
            
            pred = model(char_crop) # predict score for each class
            pred = torch.argmax(pred).item()
            
            predicted_captcha +=pred
            
        output_file.write(x + "," + predicted_captcha + "\n")

        print('Classified ' + x)

# are we using GPU or CPU?
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print("Cuda available? {}".format(torch.cuda.is_available()))
print("Device: {}".format(device))

callthis("/users/pgrad/singhr6/model_training/models", "epoch_0_lr_5e-05_batchsize_32", "/users/pgrad/singhr6/Desktop/ProcessedTestImage", "seperated_stuff.txt")
