import torch
from torchvision import models, transforms
from PIL import Image
from src.convrelu import replace_conv

replace_conv()

# Load the pre-trained AlexNet model
alexnet = models.alexnet(pretrained=True)
alexnet.eval()  # Set the model to evaluation mode

# Define image preprocessing transformations
preprocess = transforms.Compose([
    transforms.Resize(256),                # Resize the image to 256x256
    transforms.CenterCrop(224),            # Crop the image to 224x224 from the center
    transforms.ToTensor(),                 # Convert the image to a tensor
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize the tensor
])

# Load the image
img = Image.open("sample.jpeg")
img_t = preprocess(img)  # Apply the preprocessing
img_t = img_t.unsqueeze(0)  # Add a batch dimension

# Perform prediction
with torch.no_grad():  # Disable gradient calculation for inference
    output = alexnet(img_t)

# Get the predicted class index
_, predicted_idx = torch.max(output, 1)

# Load ImageNet class labels
with open("imagenet_classes.txt") as f:
    labels = [line.strip() for line in f.readlines()]

# Print the predicted class
print(f"Predicted class: {labels[predicted_idx.item()]}")
