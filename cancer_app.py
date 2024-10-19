mport streamlit as st
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

# Load the model (ensure you have the correct model file)
model = load_model('classifiaction_breast_cancer_model.h5')  # Use your Keras model file

# Create Streamlit app layout
st.set_page_config(layout="wide")

# Add login page title
st.markdown(
    """
    <div style='text-align: center; font-family: Times New Roman;'>
        <h1>User Details</h1>
    </div>
    """,
    unsafe_allow_html=True
)

# Add login page
login_name = st.text_input("Enter your name:")
login_age = st.number_input("Enter your age:", min_value=0, max_value=150, value=0)
login_gender = st.selectbox("Select your gender:", ["Male", "Female", "Other"])

# Remember login information
if st.button("Enter"):
    st.session_state['name'] = login_name
    st.session_state['age'] = login_age
    st.session_state['gender'] = login_gender

# Retrieve login information
name = st.session_state.get('name', "")
age = st.session_state.get('age', 0)
gender = st.session_state.get('gender', "")

# Display login information
st.write(f"Welcome, {name}! Age: {age}, Gender: {gender}")

# Title for the breast cancer classification app
st.markdown(
    """
    <div style='background-color: #EBDEF0; padding: 20px;'>
        <h1 style='text-align: center; font-family: Times New Roman;'>Classification Of Type Of Tumour (Breast Cancer)</h1>
    </div>
    """,
    unsafe_allow_html=True
)

# Upload image
uploaded_file = st.file_uploader("Upload a breast histopathology image", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    # Load and preprocess the image
    img = image.load_img(uploaded_file, target_size=(50, 50))  # Adjust the target size to your model's input size
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    img_array /= 255.0  # Normalize the image

    # Predict diagnosis
    if st.button('PREDICT'):
        prediction = model.predict(img_array)
        diagnosis = 'cancerous' if prediction[0][0] > 0.5 else 'non-cancerous'  # Adjust based on your model's output
        st.write(f'{name}, the tumor is {diagnosis}.')

        # Display the disclaimer message using Markdown
        st.markdown("""
            <div style='margin-top: 20px; padding: 10px; border: 1px solid #ccc; border-radius: 5px; background-color: #f8d7da; color: #721c24;'>
                <strong>Disclaimer:</strong> The prediction provided on this page is for assessment purposes only and may not always be accurate. It is recommended to consult a healthcare professional for accurate diagnosis and advice.
            </div>
            """,
            unsafe_allow_html=True
        )

# Disclaimer message
st.markdown("""
    <div style='margin-top: 20px; padding: 10px; border: 1px solid #ccc; border-radius: 5px; background-color: #f8d7da; color: #721c24;'>
        <strong>Disclaimer:</strong> The prediction provided on this page is for assessment purposes only and may not always be accurate. It is recommended to consult a healthcare professional for accurate diagnosis and advice.
    </div>
    """,
    unsafe_allow_html=True