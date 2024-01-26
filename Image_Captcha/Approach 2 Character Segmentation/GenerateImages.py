# -*- coding: utf-8 -*-
"""
Created on Wed Oct 11 17:43:47 2023

"""

###########################################################
## ReadME                  ################################
## symbols - Complete path to the symbol file           ###
## output_dir - Complete path where to generate captchas ##
## eamon_font_path - Complete path to the font file      ##
###########################################################

import os
import numpy
import random
import cv2
import captcha.image
import concurrent.futures  # Import the concurrent.futures for multithreading

from PIL import ImageFont, Image, ImageDraw

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

width = 128
height = 64
symbols = '/users/pgrad/singhr6/model_training/symbols.txt'
output_dir = r'/users/pgrad/singhr6/Desktop/ValidateImage2' 
count = 40000
max_length = 1
eamon_font_path = []
eamon_font_path.append(r'/users/pgrad/singhr6/model_training/eamonexpbold.woff.ttf')

os.chmod(output_dir, 0o777)
captcha_generator = captcha.image.ImageCaptcha(width=width, height=height, fonts=eamon_font_path)

symbols_file = open(symbols, 'r')
captcha_symbols = symbols_file.readline().strip()
symbols_file.close()

print("Generating captchas with symbol set {" + captcha_symbols + "}")

if not os.path.exists(output_dir):
    print("Creating output directory " + output_dir)
    os.makedirs(output_dir)

def generate_captcha(random_str):
    random_str = random_str.strip()
    random_str = random_str + "_" + str(random.randint(0, count*2))
    image_path = os.path.join(output_dir, encode_special_characters(random_str) +'.png')
    
    if random_str:
        image = numpy.array(captcha_generator.generate_image(decode_special_characters(random_str)))
        
        if ("ValidateImage2" in image_path):
            try:
                cv2.imwrite(image_path, image)
            except Exception as e:
                print(f"Not able to write the Image: {e}")
        else:
            print("ValidateImage not found in image_path : " + image_path)
    else:
        print("Random Str empty or not considered : " + random_str)

random_strings = [''.join([random.choice(captcha_symbols) for _ in range(random.randint(1, max_length))]) for _ in range(count)]

# Use ThreadPoolExecutor to execute the generate_captcha function with multiple threads
with concurrent.futures.ThreadPoolExecutor() as executor:
    executor.map(generate_captcha, random_strings)

files = os.listdir(output_dir)

# Filter files to keep only those with .png extension
png_files = [file for file in files if file.lower().endswith('.png')]

# Remove non-PNG files
for file in files:
    file_path = os.path.join(output_dir, file)
    if file not in png_files and os.path.isfile(file_path):
        os.remove(file_path)
        print("File deleted : " + file_path)
