import os
import shutil

def clean_dataset(base_path, num_classes=20, num_images_per_class=50):
    # Define paths
    train_path = os.path.join(base_path, 'train')
    val_path = os.path.join(base_path, 'val')

    # Function to keep only the specified number of classes and images
    def process_directory(directory, num_classes, num_images_per_class):
        classes = sorted(os.listdir(directory))
        
        # Keep only the first `num_classes` classes
        for class_idx, class_name in enumerate(classes):
            if class_idx >= num_classes:
                shutil.rmtree(os.path.join(directory, class_name))
                continue
            
            class_path = os.path.join(directory, class_name)
            images = sorted(os.listdir(class_path))
            
            # Keep only the first `num_images_per_class` images
            for image_idx, image_name in enumerate(images):
                if image_idx >= num_images_per_class:
                    os.remove(os.path.join(class_path, image_name))
                    
            # Remove the class directory if it becomes empty
            if not os.listdir(class_path):
                shutil.rmtree(class_path)
    
    # Process both training and validation datasets
    process_directory(train_path, num_classes, num_images_per_class)
    process_directory(val_path, num_classes, num_images_per_class)

# Call the function with your dataset path
clean_dataset('Project/tiny-imagenet-20Class-50Data')
