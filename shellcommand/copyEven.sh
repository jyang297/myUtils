#!/bin/bash

# Directory containing the images
src_dir="frames"

# Directory where odd-numbered images will be copied
dest_dir="odd_frames"

# Create destination directory if it doesn't exist
mkdir -p "$dest_dir"

# Loop through all png files in the source directory
for file in "$src_dir"/frame_*.png; do
    echo "Processing $file"  # Debug output to see which file is being processed

    # Extract the number part of the filename
    filename=$(basename "$file")
    number=${filename#frame_}
    number=${number%.png}

    # Convert number to decimal to avoid octal interpretation
    decimal_number=$((10#$number))

    echo "Number extracted: $decimal_number"  # Debug output to check the extracted number

    # Check if the number is odd
    if [ $((decimal_number % 2)) -ne 0 ]; then
        echo "Copying $file to $dest_dir"  # Debug output to confirm file copy
        cp "$file" "$dest_dir"
    else
        echo "$file is not odd"  # Debug output for even files
    fi
done
