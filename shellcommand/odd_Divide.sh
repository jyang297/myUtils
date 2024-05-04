#!/bin/bash

# Directory containing the odd-numbered images
src_dir="odd_frames"

# Base directory where subdirectories will be created
base_dest_dir="odd_frames_divided"

# Number of images per subfolder before cycling to the next
images_per_folder=21

# Create base directory if it doesn't exist
mkdir -p "$base_dest_dir"

# Create 8 subdirectories
for i in {1..8}; do
    mkdir -p "$base_dest_dir/subfolder_$i"
done

# Initialize a counter to distribute images across subfolders
counter=1

# Index to track the number of processed images
index=1

# Loop through all png files in the source directory, sorted numerically
for file in $(ls "$src_dir"/frame_*.png | sort -V); do
    # Calculate the subfolder number based on the current index
    folder_number=$(((index - 1) / images_per_folder % 8 + 1))

    echo "Distributing $file to subfolder_$folder_number"  # Debug output

    # Copy file to the current subfolder
    cp "$file" "$base_dest_dir/subfolder_$folder_number"

    # Increment index
    ((index++))
done
