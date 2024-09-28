import streamlit as st

st.title("OCR Application")
st.write("Upload an image to extract text in Hindi and English.")

# Upload image
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Save the uploaded file temporarily
    image_path = "/kaggle/temp/uploaded_image.png"
    with open(image_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    # Perform OCR
    extracted_text = ocr_image(image_path)

    st.subheader("Extracted Text:")
    st.write(extracted_text)

    # Keyword search functionality
    keyword = st.text_input("Enter a keyword to search:")
    if keyword:
        if keyword in extracted_text:
            highlighted_text = extracted_text.replace(keyword, f"**{keyword}**")
            st.markdown("### Search Result:")
            st.markdown(highlighted_text)
        else:
            st.write("Keyword not found.")
