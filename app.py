import streamlit as st
from PIL import Image
import torch
from transformers import AutoProcessor, VisionEncoderDecoderModel
token="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"
# Load the ColPali implementation and Qwen2-VL model
ocr_model_name = "stepfun-ai/GOT-OCR2_0"  # Update with the correct Hugging Face model path if needed
processor = AutoProcessor.from_pretrained(ocr_model_name,token=token)
ocr_model = VisionEncoderDecoderModel.from_pretrained(ocr_model_name,token=token)

def extract_text(image):
    """Extracts text from the given image using the ColPali OCR model."""
    # Prepare the image for the model
    inputs = processor(images=image, return_tensors="pt")

    # Perform OCR
    with torch.no_grad():
        output = ocr_model.generate(**inputs)

    # Decode the generated text
    extracted_text = processor.decode(output[0], skip_special_tokens=True)
    return extracted_text

def main():
    st.title("OCR Web Application with Qwen2-VL")
    st.write("This application extracts text from an uploaded image and allows interaction with Qwen2-VL.")

    # Upload image
    uploaded_file = st.file_uploader("Choose an image file", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        # Open the uploaded image
        image = Image.open(uploaded_file)
        
        # Display the uploaded image
        st.image(image, caption="Uploaded Image", use_column_width=True)

        # Extract text from the image
        with st.spinner('Extracting text...'):
            extracted_text = extract_text(image)

        # Display the extracted text
        st.subheader("Extracted Text:")
        st.text_area(label="", value=extracted_text, height=250)

        # Additional interaction with Qwen2-VL
        st.subheader("Interact with Qwen2-VL")
        user_input = st.text_input("Ask something about the extracted text:")

        if user_input:
            # Here you can add functionality to process the user input with the Qwen2-VL model
            # For demonstration purposes, we can just display the user input
            response = f"You asked: {user_input}. Here’s some processed response based on the extracted text."
            st.write(response)

if __name__ == "__main__":
    main()
