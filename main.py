import torch
from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image
import os

# Load the BLIP processor and model
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base").to("cuda" if torch.cuda.is_available() else "cpu")

def ocr_image(image_path):
    """
    Perform OCR using BLIP (PyTorch) to extract text from an image.
    
    Args:
        image_path (str): Path to the image file.
    
    Returns:
        str: Extracted text from the image.
    """
    # Open and preprocess the image
    image = Image.open(image_path).convert("RGB")
    inputs = processor(image, return_tensors="pt").to("cuda" if torch.cuda.is_available() else "cpu")

    # Perform inference
    with torch.no_grad():
        generated_ids = model.generate(**inputs)
        extracted_text = processor.decode(generated_ids[0], skip_special_tokens=True)

    return extracted_text

# Example usage: Upload an image to Kaggle notebook and specify the path
image_path = "/kaggle/input/testimage/Screenshot 2024-09-21 034510.png"  # Replace with your image path

# Perform OCR on the uploaded image
extracted_text = ocr_image(image_path)

# Display the extracted text
print("Extracted Text:\n", extracted_text)
