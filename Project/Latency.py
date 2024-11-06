import torch
import torchvision.models as models
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
import time
import os

# Print the current directory
current_dir = os.getcwd()
print("Current Directory:", current_dir)




# 1. Load the Model Structure and State Dict
model_path = 'Project/ResNet-without_FC/best_model.pth'  # Replace with your actual model path
model = models.resnet18()  # Adjust this if it's a custom model
# Modify the final layer to match the number of classes (200 for Tiny ImageNet)
model.fc = torch.nn.Linear(model.fc.in_features, 200)  # Adjusting for Tiny ImageNet's 200 classes

model.load_state_dict(torch.load(model_path))  # Load the weights into the model
model.eval()  # 

# 2. Prepare the Dataset
# Assuming Tiny ImageNet images are 64x64 in RGB format
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalization for pretrained models
])

# Load validation dataset from a directory
val_dataset_path = 'Project/tiny-imagenet-200Class-500Data/'  # Replace with your dataset path
val_dataset = ImageFolder(root=val_dataset_path, transform=transform)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

# 3. Measure Inference Latency
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

latency_times = []

with torch.no_grad():  # Disable gradient calculations for inference
    for images, _ in val_loader:
        images = images.to(device)
        
        # Measure the time for a single forward pass
        start_time = time.time()
        outputs = model(images)
        end_time = time.time()
        
        latency = end_time - start_time  # Time for the batch
        latency_times.append(latency)

# 4. Calculate Average Latency
average_latency = sum(latency_times) / len(latency_times)
print(f"Average Latency per Batch: {average_latency:.10f} seconds")

# Optional: Calculate Latency per Image
batch_size = val_loader.batch_size
average_latency_per_image = average_latency / batch_size
print(f"Average Latency per Image: {average_latency_per_image:.10f} seconds")
