#!/bin/bash



mkdir -p "$base_dest_dir"
src_dir="frames"
base_dest_dir="frames_overlapping"

# Number of images per folder, with overlap
images_per_folder=19
# Initialize counters
folder_number=1 # Do i need to start with 0?
start_image=1




# Determine the total number of images
total_images=$(ls "$src_dir"/frame_*.png | wc -l)

# Loop until we reach or exceed the total number of images
while [ $start_image -le $total_images ]; do
    end_image=$((start_image + images_per_folder))  # calculate end image for current folder

    # Create a new subfolder
    mkdir -p "$base_dest_dir/subfolder_$folder_number"

    for ((i=start_image; i<end_image; i++)); do
        file=$(printf "$src_dir/frame_%04d.png" $i)
        if [ -f "$file" ]; then
            echo "Copying $file to subfolder_$folder_number"  # Debug output
            cp "$file" "$base_dest_dir/subfolder_$folder_number"
        fi
    done

    # Update start_image for the next batch. Overlap here.
    start_image=$((start_image + images_per_folder - 1))
    folder_number=$((folder_number + 1))
done
