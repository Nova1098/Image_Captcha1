# -*- coding: utf-8 -*-
"""
Created on Fri Oct 13 03:33:26 2023

@author: rahul
"""

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--learning-rate', type = float, action = 'store', default = 0.00005)
parser.add_argument('--num-epochs', type = int, action = 'store', default = 50)
parser.add_argument('--weight-decay', type = float, action = 'store', default = 0.98)
parser.add_argument('--batch-size', type = int, action = 'store', default = 32)
args = parser.parse_args()
