import os
import cv2
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F

# Define the CNNtoRNN model architecture (same as in your original code)
class CNNtoRNN(nn.Module):
    def __init__(self, num_of_characters):
        super(CNNtoRNN, self).__init__()
        # Convolutional layers
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        # Max pooling layers
        self.maxpool1 = nn.MaxPool2d(kernel_size=2)
        self.maxpool2 = nn.MaxPool2d(kernel_size=2)
        self.maxpool3 = nn.MaxPool2d(kernel_size=(1, 2))
        # Dropout layers
        self.dropout1 = nn.Dropout(0.3)
        self.dropout2 = nn.Dropout(0.3)
        # Fully connected layer
        self.fc1 = nn.Linear(128 * 8, 64)
        # Bidirectional LSTM layers
        self.lstm1 = nn.LSTM(64, 256, bidirectional=True, batch_first=True)
        self.lstm2 = nn.LSTM(512, 256, bidirectional=True, batch_first=True)
        # Final fully connected layer
        self.fc2 = nn.Linear(512, num_of_characters)

    def forward(self, x):
        x = self.maxpool1(F.relu(self.bn1(self.conv1(x))))
        x = self.maxpool2(F.relu(self.bn2(self.conv2(x))))
        x = self.dropout1(x)
        x = self.maxpool3(F.relu(self.bn3(self.conv3(x))))
        x = self.dropout2(x)
        # Reshape
        batch, channels, height, width = x.size()
        x = x.permute(0, 2, 1, 3).contiguous().view(batch, height, channels * width)
        # Fully connected layer
        x = F.relu(self.fc1(x))
        # Bidirectional LSTM layers
        x, _ = self.lstm1(x)
        x, _ = self.lstm2(x)
        # Output layer
        x = self.fc2(x)
        x = F.log_softmax(x, dim=2)
        return x

# PreprocessTransform class (same as in your original code)
class PreprocessTransform:
    def __call__(self, img):
        img = np.array(img)
        (h, w) = img.shape
        final_img = np.ones([64, 256]) * 255  # blank white image
        # crop
        if w > 256:
            img = img[:, :256]
        if h > 64:
            img = img[:64, :]
        final_img[:h, :w] = img
        final_img = cv2.rotate(final_img, cv2.ROTATE_90_CLOCKWISE)
        # Convert to PyTorch tensor and normalize to [0, 1]
        final_img = torch.tensor(final_img, dtype=torch.float32) / 255.0
        final_img = final_img.unsqueeze(0)  # Add channel dimension
        return final_img

# Decoding function (same as in your original code)
def greedy_decoder(output, labels):
    output = output.cpu().numpy()
    arg_maxes = np.argmax(output, axis=2)
    decodes = []
    for i in range(arg_maxes.shape[1]):
        args = arg_maxes[:, i]
        decode = []
        for j in range(args.shape[0]):
            index = args[j]
            if index != 0:
                decode.append(labels[index])
        decodes.append(decode)
    return decodes

# Labels (same as in your original code)
labels = ['<BLANK>', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', '-', "'", '`', ' ']

# Function to evaluate a single image
def evaluate_single_image(image_path, model_path):
    # Load the pre-trained model
    model = torch.load(model_path, map_location=torch.device('cpu')) 
    model.eval()  # Set the model to evaluation mode
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    
    # Preprocess the image
    transform = PreprocessTransform()
    image = Image.open(image_path).convert('L')
    image = transform(image)
    image = image.unsqueeze(0).to(device)  # Add batch dimension and move to device

    # Make prediction
    with torch.no_grad():
        outputs = model(image)
        outputs = outputs.permute(1, 0, 2)
        outputs = F.log_softmax(outputs, dim=2)
        decoded_output = greedy_decoder(outputs, labels)
        prediction = ''.join(decoded_output[0])

    return prediction

# Example usage
image_path = 'TEST_0067.jpg'  # Path to your test image
model_path = 'model_60 (1).pth'  # Path to your saved model
prediction = evaluate_single_image(image_path, model_path)
print(f"Prediction for {image_path}: {prediction}")