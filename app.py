# pip install streamlit torch torchvision transformers Pillow
import streamlit as st
from PIL import Image
import torch
from transformers import VisionEncoderDecoderModel, ViTImageProcessor, AutoTokenizer

# Load the model and tokenizer
model_name = "Salesforce/codet5-base"  # Change this to the correct model
tokenizer = AutoTokenizer.from_pretrained(model_name)
processor = ViTImageProcessor.from_pretrained(model_name)
model = VisionEncoderDecoderModel.from_pretrained(model_name)

def extract_text(image):
    """Extracts text from the given image using the chosen OCR model."""
    # Prepare image for model
    inputs = processor(images=image, return_tensors="pt")

    # Perform OCR
    output = model.generate(**inputs)

    # Decode the generated text
    return tokenizer.decode(output[0], skip_special_tokens=True)

def main():
    st.title("OCR Web Application")
    st.write("This application extracts text from an uploaded image and supports both Hindi and English languages.")

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

        # Search functionality
        st.subheader("Search in Extracted Text")
        search_keyword = st.text_input("Enter a keyword to search")

        if search_keyword:
            # Check if the keyword exists in the extracted text
            if search_keyword.lower() in extracted_text.lower():
                st.success(f"Keyword '{search_keyword}' found!")
                highlighted_text = extracted_text.replace(search_keyword, f"**{search_keyword}**")
                st.write(highlighted_text, unsafe_allow_html=True)
            else:
                st.error(f"Keyword '{search_keyword}' not found.")

if __name__ == "__main__":
    main()
