import os
import shutil

# Set the path to your dataset
dataset_path = "Project/tiny-imagenet-50/train"

# Loop through each class directory
for class_dir in os.listdir(dataset_path):
    class_path = os.path.join(dataset_path, class_dir)
    
    if os.path.isdir(class_path):
        images_dir = os.path.join(class_path, 'images')
        
        # Check if 'images/' directory exists
        if os.path.exists(images_dir):
            # Move each image from 'images/' to the class directory
            for img_file in os.listdir(images_dir):
                img_path = os.path.join(images_dir, img_file)
                if os.path.isfile(img_path):
                    shutil.move(img_path, class_path)  # Move the image to class directory
            
            # Remove the now empty 'images/' directory
            os.rmdir(images_dir)
        
        # Remove all .txt files in the class directory
        for file in os.listdir(class_path):
            if file.endswith('.txt'):
                os.remove(os.path.join(class_path, file))

print("Process completed successfully!")
