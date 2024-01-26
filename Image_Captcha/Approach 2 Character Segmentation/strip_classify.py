# -*- coding: utf-8 -*-
"""
Created on Tue Oct 24 19:43:17 2023

"""

# Open the input file in read mode
input_file_path = 'stuff.txt'  # Specify the path to your input file

# Create a list to store data in CSV format
csv_data = []

with open(input_file_path, 'r') as input_file:
    # Read lines from the input file and remove trailing ")" symbols
    lines = [line.rstrip(')\n') for line in input_file.readlines()]

# Read lines from the input file and process them
with open(input_file_path, 'r') as input_file:
    for line in lines:
        line = line.replace(".jpg", ".png")
        # Split the line at commas, ignoring leading and trailing spaces
        parts = [((part.strip().replace(",", ""))) for part in line.strip().split(',', 1)]
        # Handle empty values and create a CSV row
        csv_row = ','.join(parts)
        # Append the CSV row to the CSV data list
        csv_data.append(csv_row)

# Write data into the CSV file
output_csv_path = 'output_file.csv'  # Specify the path to your output CSV file
with open(output_csv_path, 'w') as output_csv_file:
    for row in csv_data:
        # Write each CSV row as a new line in the CSV file
        output_csv_file.write("%s\n" % row)

print("Data converted to CSV format and saved in output_file.csv.")
