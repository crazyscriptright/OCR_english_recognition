# Image OCR Demo

This project is a Streamlit-based application for Optical Character Recognition (OCR). It allows users to upload an image and extracts text from it using a deep learning model.

## Features

- **Image Upload**: Upload images in `.jpg`, `.jpeg`, or `.png` formats.
- **OCR Processing**: Processes the uploaded image and extracts text using a CNN-to-RNN model.
- **Preprocessing**: Includes image preprocessing such as contrast enhancement, adaptive thresholding, and resizing.
- **Streamlit UI**: Interactive web interface for easy usage.

## How It Works

1. **Image Upload**: Users upload an image via the Streamlit interface.
2. **Preprocessing**: The image is preprocessed to enhance OCR accuracy.
3. **Model Inference**: A trained CNN-to-RNN model predicts the text in the image.
4. **Output**: The recognized text is displayed on the web interface.

## Requirements

- Python 3.8+
- Streamlit
- PyTorch
- OpenCV
- NumPy
- Pillow

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/your-repo/image-ocr-demo.git
   cd image-ocr-demo
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Run the application:
   ```bash
   streamlit run googleimgtest.py
   ```

## Model

The application uses a CNN-to-RNN model for OCR. Ensure the model file (`model_60 (1).pth`) is in the project directory.

## File Structure

- `googleimgtest.py`: Main application script.
- `model_60 (1).pth`: Pre-trained model file.
- `requirements.txt`: List of dependencies.

## Example

1. Upload an image:
   ![Upload Example](example-upload.png)

2. View the OCR output:
   ![OCR Output](example-output.png)

## License

This project is licensed under the MIT License.

## Acknowledgments

- PyTorch for the deep learning framework.
- Streamlit for the web interface.
- OpenCV for image preprocessing.
