import os
import cv2
from utils import validate_image_extension

def scale_images(input_folder, output_folder):
    min_dimension = float('inf')  # Initialize min_dimension to positive infinity
    
    # Create the output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)
    
    # Iterate over files in the input folder
    for filename in os.listdir(input_folder):
        if validate_image_extension(os.path.join(input_folder, filename)):
            input_image_path = os.path.join(input_folder, filename)
            img = cv2.imread(input_image_path)
            print(img.shape)
            height, width, _ = img.shape
            min_dimension = min(min_dimension, min(width, height))

    # Scale down other images to match the lesser dimension
    for filename in os.listdir(input_folder):
        if validate_image_extension(os.path.join(input_folder, filename)):
            input_image_path = os.path.join(input_folder, filename)
            img = cv2.imread(input_image_path)
            height, width, _ = img.shape
            print(height, width, _)
            if width != height:
                ratio = min_dimension / min(width, height)
                new_width = int(width * ratio)
                new_height = int(height * ratio)
                print(new_width, new_height)
                resized_img = cv2.resize(img, (new_width, new_height))
                
                # Write the scaled-down image to the output folder
                output_image_path = os.path.join(output_folder, filename)
                cv2.imwrite(output_image_path, resized_img)

# Example usage
input_folder = 'src/rough/grayscale'
output_folder = 'src/outpost/'
scale_images(input_folder, output_folder)
