import streamlit as st
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

# Define the CNNtoRNN model architecture (same as before)
class CNNtoRNN(nn.Module):
    def __init__(self, num_of_characters):
        super(CNNtoRNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.maxpool1 = nn.MaxPool2d(kernel_size=2)
        self.maxpool2 = nn.MaxPool2d(kernel_size=2)
        self.maxpool3 = nn.MaxPool2d(kernel_size=(1, 2))
        self.dropout1 = nn.Dropout(0.3)
        self.dropout2 = nn.Dropout(0.3)
        self.fc1 = nn.Linear(128 * 8, 64)
        self.lstm1 = nn.LSTM(64, 256, bidirectional=True, batch_first=True)
        self.lstm2 = nn.LSTM(512, 256, bidirectional=True, batch_first=True)
        self.fc2 = nn.Linear(512, num_of_characters)

    def forward(self, x):
        x = self.maxpool1(F.relu(self.bn1(self.conv1(x))))
        x = self.maxpool2(F.relu(self.bn2(self.conv2(x))))
        x = self.dropout1(x)
        x = self.maxpool3(F.relu(self.bn3(self.conv3(x))))
        x = self.dropout2(x)
        batch, channels, height, width = x.size()
        x = x.permute(0, 2, 1, 3).contiguous().view(batch, height, channels * width)
        x = F.relu(self.fc1(x))
        x, _ = self.lstm1(x)
        x, _ = self.lstm2(x)
        x = self.fc2(x)
        x = F.log_softmax(x, dim=2)
        return x


class PreprocessTransform:
    def __call__(self, img):
        img = np.array(img)
        (h, w) = img.shape
        final_img = np.ones([64, 256]) * 255
        if w > 256:
            img = img[:, :256]
        if h > 64:
            img = img[:64, :]
        final_img[:h, :w] = img
        final_img = cv2.rotate(final_img, cv2.ROTATE_90_CLOCKWISE)
        final_img = torch.tensor(final_img, dtype=torch.float32) / 255.0
        final_img = final_img.unsqueeze(0)
        return final_img


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


labels = ['<BLANK>', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', '-', "'", '`', ' ']


def evaluate_image(image, model):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    transform = PreprocessTransform()
    model.to(device)
    model.eval()
    image = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        outputs = model(image)
        outputs = outputs.permute(1, 0, 2)
        decoded_output = greedy_decoder(outputs, labels)
        prediction = ''.join(decoded_output[0])
    return prediction


# Streamlit UI
st.set_page_config(page_title="OCR Demo", layout="wide")

st.title("Image OCR Demo")
st.markdown(
    """
    Upload an image to see the OCR output. 
    The model will process the image and display the recognized text below.
    """
)

uploaded_file = st.file_uploader("Choose an image file", type=["jpg", "jpeg", "png"])
if uploaded_file:
    # Display the uploaded image
    image = Image.open(uploaded_file).convert("L")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Load the model
    model_path = "model_60 (1).pth"  # Update with your model path
    model = torch.load(model_path, map_location=torch.device("cpu"))

    # Perform OCR
    prediction = evaluate_image(image, model)

    # Display the result
    st.markdown("### OCR Output:")
    st.write(f"**{prediction}**")
