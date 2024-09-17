import os
import shutil
import os

# Print the current working directory
current_dir = os.getcwd()
print(f"Current directory: {current_dir}")

# Paths to your data
val_dir = 'Project/tiny-imagenet-200/val'  # Change this to your actual path
val_annotations = os.path.join(val_dir, 'val_annotations.txt')
images_dir = os.path.join(val_dir, 'images')

# Step 1: Create a dictionary mapping image names to class labels
image_class_map = {}
with open(val_annotations, 'r') as f:
    for line in f.readlines():
        parts = line.strip().split()
        image_file = parts[0]
        class_label = parts[1]
        image_class_map[image_file] = class_label

# Step 2: Create directories for each class in the val directory
for class_label in set(image_class_map.values()):
    class_dir = os.path.join(val_dir, class_label)
    if not os.path.exists(class_dir):
        os.makedirs(class_dir)

# Step 3: Move images to the corresponding class directory
for image_file, class_label in image_class_map.items():
    src = os.path.join(images_dir, image_file)
    dst = os.path.join(val_dir, class_label, image_file)
    shutil.move(src, dst)

# Optional: Remove the 'images' directory after moving
shutil.rmtree(images_dir)

print("Validation data organized successfully.")
