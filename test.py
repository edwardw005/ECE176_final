import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import os

# Define the CNN model structure - exactly matching the training model
class RockPaperScissorsCNN(nn.Module):
    def __init__(self):
        super(RockPaperScissorsCNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(64 * 32 * 32, 128)
        self.fc2 = nn.Linear(128, 4)  # 4 output classes: rock, paper, scissors, unknown  
    
    def forward(self, x):
        # Using F.relu to match training code exactly
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(x.shape[0], -1)  # Flatten the tensor
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Load the trained model
model_path = "rps_model.pth"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
model = RockPaperScissorsCNN().to(device)
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()  # Set to evaluation mode

# Define the transformation for test images - MUST MATCH training transforms
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))  # Added normalization to match training
])

# Class mapping - make sure this matches your training classes
classes = ['rock', 'paper', 'scissors', 'unknown']
print(f"Class mapping: {dict(enumerate(classes))}")

# Function to predict a single image
def predict_image(image_path):
    """
    Predict the class of an image using the trained model
    """
    # Load and transform the image
    image = Image.open(image_path).convert('RGB')
    original_image = image.copy()  # Keep an unmodified copy for display
    image_tensor = transform(image).unsqueeze(0).to(device)
    
    # Get model prediction
    with torch.no_grad():
        output = model(image_tensor)
        _, predicted = torch.max(output, 1)
        probabilities = F.softmax(output, dim=1)[0]  # Using F.softmax to match training
    
    # Convert prediction to class name
    predicted_class = classes[predicted.item()]
    
    # Create results dictionary
    results = {
        'class': predicted_class,
        'probabilities': {classes[i]: float(probabilities[i]) * 100 for i in range(len(classes))}
    }
    
    # Display the results
    display_results(original_image, results)
    
    return results

# Function to display results
def display_results(image, prediction_results):
    """
    Display the image and prediction results
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
    
    # Display the image
    ax1.imshow(image)
    ax1.set_title('Test Image')
    ax1.axis('off')
    
    # Display the prediction probabilities as a bar chart
    classes = list(prediction_results['probabilities'].keys())
    probabilities = list(prediction_results['probabilities'].values())
    
    colors = ['blue' if cls != prediction_results['class'] else 'green' for cls in classes]
    
    y_pos = np.arange(len(classes))
    ax2.barh(y_pos, probabilities, color=colors)
    ax2.set_yticks(y_pos)
    ax2.set_yticklabels(classes)
    ax2.set_xlim(0, 100)
    ax2.set_title('Prediction Probabilities (%)')
    ax2.set_xlabel('Probability (%)')
    
    plt.tight_layout()
    plt.show()
    
    # Print the results
    print(f"Predicted class: {prediction_results['class']}")
    print("\nProbabilities:")
    for cls, prob in prediction_results['probabilities'].items():
        print(f"{cls}: {prob:.2f}%")

# Function to test multiple images in a directory
def predict_directory(directory_path):
    """
    Test all JPEG images in a directory
    """
    # Get all jpg/jpeg files in the directory
    image_files = [os.path.join(directory_path, f) for f in os.listdir(directory_path) 
                  if f.lower().endswith(('.jpg', '.jpeg', '.png'))]  # Added PNG support
    
    if not image_files:
        print("No image files found in the directory.")
        return
    
    print(f"Found {len(image_files)} images.")
    
    # Process each image
    results = {}
    for image_path in image_files:
        print(f"\nProcessing: {os.path.basename(image_path)}")
        results[image_path] = predict_image(image_path)
    
    return results

# Example usage:
if __name__ == "__main__":
    # Test a single image
    # predict_image("path_to_test_image.jpg")
    
    # Or test a directory of images
    # predict_directory("path_to_test_directory")
    
    # Interactive mode - ask for input
    while True:
        print("\nOptions:")
        print("1. Test a single image")
        print("2. Test all images in a directory")
        print("3. Exit")
        choice = input("Enter your choice (1/2/3): ")
        
        if choice == '1':
            image_path = input("Enter the path to the image: ")
            if os.path.exists(image_path):
                predict_image(image_path)
            else:
                print("Invalid path!")
        elif choice == '2':
            directory_path = input("Enter the directory path: ")
            if os.path.isdir(directory_path):
                predict_directory(directory_path)
            else:
                print("Invalid directory!")
        elif choice == '3':
            break
        else:
            print("Invalid choice!")
