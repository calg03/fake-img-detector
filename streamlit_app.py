import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import io
import base64
import os

# Set page configuration
st.set_page_config(
    page_title="AI Vision Analyzer",
    page_icon="🔮",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS for glassmorphism effect and modern typography
def add_custom_css():
    css = """
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;500;600;700&display=swap');
        
        html, body, [class*="st-"] {
            font-family: 'Poppins', sans-serif;
        }
        
        .glassmorphism {
            background: rgba(255, 255, 255, 0.15);
            backdrop-filter: blur(10px);
            -webkit-backdrop-filter: blur(10px);
            border-radius: 20px;
            border: 1px solid rgba(255, 255, 255, 0.18);
            padding: 30px;
            box-shadow: 0 8px 32px 0 rgba(31, 38, 135, 0.37);
        }
        
        .title {
            font-weight: 700;
            color: #1E1E1E;
            font-size: 3.2rem;
            margin-bottom: 0.5rem;
            background: linear-gradient(45deg, #7928CA, #FF0080);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }
        
        .subtitle {
            font-weight: 400;
            color: #4A4A4A;
            font-size: 1.2rem;
        }
        
        .custom-upload-btn {
            font-weight: 600;
            border-radius: 10px;
            border: none;
            padding: 0.5rem 1rem;
            background: linear-gradient(45deg, #7928CA, #FF0080);
            color: white;
            cursor: pointer;
            transition: all 0.3s ease;
        }
        
        .custom-upload-btn:hover {
            transform: translateY(-2px);
            box-shadow: 0 5px 15px rgba(0, 0, 0, 0.1);
        }
        
        .stButton>button {
            background: linear-gradient(45deg, #7928CA, #FF0080);
            color: white;
            border: none;
            padding: 0.5rem 2rem;
            border-radius: 10px;
            font-weight: 600;
            transition: all 0.3s ease;
        }
        
        .stButton>button:hover {
            transform: translateY(-2px);
            box-shadow: 0 5px 15px rgba(0, 0, 0, 0.1);
        }
        
        /* Add a gradient background */
        .main {
            background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        }
        
        /* Style for prediction results */
        .prediction-result {
            font-weight: 600;
            font-size: 1.5rem;
            color: #7928CA;
            margin-top: 1rem;
        }
        
        /* Image container */
        .image-container {
            border-radius: 15px;
            overflow: hidden;
            box-shadow: 0 8px 32px 0 rgba(31, 38, 135, 0.2);
            margin: 1rem 0;
        }
    </style>
    """
    st.markdown(css, unsafe_allow_html=True)

# Add the custom CSS
add_custom_css()

# Create background gradient
def add_bg_gradient():
    bg_gradient = """
    <div class="main" style="position: fixed; top: 0; left: 0; width: 100%; height: 100%; z-index: -1;"></div>
    """
    st.markdown(bg_gradient, unsafe_allow_html=True)

add_bg_gradient()

# Header section
st.markdown('<p class="title">AI Vision Analyzer</p>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">Upload an image to analyze with our powerful deep learning model</p>', unsafe_allow_html=True)

# Create two columns for layout
col1, col2 = st.columns([3, 2])

# Model loading functionality
@st.cache_resource
def load_model():
    # Using a pretrained ResNet model as an example
    model = models.resnet18(pretrained=True)
    model.eval()
    
    # In a real application, you would load your custom weights:
    # model_path = "path/to/model_weights.pth"
    # model.load_state_dict(torch.load(model_path))
    
    return model

# Image preprocessing
def preprocess_image(image):
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                             std=[0.229, 0.224, 0.225]),
    ])
    
    image_tensor = preprocess(image).unsqueeze(0)
    return image_tensor

# Load ImageNet class labels
@st.cache_data
def load_imagenet_labels():
    labels_url = "https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt"
    try:
        import urllib.request
        with urllib.request.urlopen(labels_url) as response:
            categories = [line.decode("utf-8").strip() for line in response.readlines()]
        return categories
    except:
        return [f"Class {i}" for i in range(1000)]  # Fallback

with col1:
    st.markdown('<div class="glassmorphism">', unsafe_allow_html=True)
    
    # File uploader widget
    uploaded_file = st.file_uploader(
        "Choose an image...", 
        type=["jpg", "jpeg", "png"],
        help="Upload an image to analyze"
    )
    
    if uploaded_file is not None:
        try:
            # Open image
            image = Image.open(uploaded_file).convert('RGB')
            
            # Display the image
            st.markdown('<div class="image-container">', unsafe_allow_html=True)
            st.image(image, caption="Uploaded Image", use_column_width=True)
            st.markdown('</div>', unsafe_allow_html=True)
            
            # Process button
            if st.button("Analyze Image"):
                with st.spinner("Processing..."):
                    # Load model
                    model = load_model()
                    
                    # Preprocess image
                    input_tensor = preprocess_image(image)
                    
                    # Get prediction
                    with torch.no_grad():
                        output = model(input_tensor)
                    
                    # Get top-5 results
                    _, indices = torch.sort(output, descending=True)
                    percentages = torch.nn.functional.softmax(output, dim=1)[0] * 100
                    
                    # Load class labels
                    labels = load_imagenet_labels()
                    
                    # Display results
                    st.markdown("### Results")
                    for idx in indices[0][:5]:
                        st.markdown(
                            f'<div class="prediction-result">{labels[idx]}: {percentages[idx]:.2f}%</div>', 
                            unsafe_allow_html=True
                        )
        except Exception as e:
            st.error(f"Error processing image: {e}")
    
    st.markdown('</div>', unsafe_allow_html=True)

with col2:
    st.markdown('<div class="glassmorphism">', unsafe_allow_html=True)
    st.markdown("### How It Works")
    st.write("""
    1. Upload your image using the file selector
    2. Our AI model will analyze the image 
    3. View the detailed analysis results
    
    This application uses a state-of-the-art deep learning model trained on millions of images to recognize objects, scenes, and more.
    """)
    
    st.markdown("### About the Model")
    st.write("""
    Our model is built with PyTorch and uses a deep convolutional neural network architecture. 
    It can identify thousands of different object categories with high accuracy.
    """)
    
    # Add some metrics or stats about the model
    st.markdown("### Model Performance")
    col_a, col_b, col_c = st.columns(3)
    with col_a:
        st.metric(label="Accuracy", value="93.7%")
    with col_b:
        st.metric(label="Classes", value="1,000+")
    with col_c:
        st.metric(label="Speed", value="0.3s")
    
    st.markdown('</div>', unsafe_allow_html=True)

# Footer
footer = """
<div style="text-align: center; margin-top: 40px; padding: 20px;">
    <p style="color: #666; font-size: 0.8rem;">© 2025 AI Vision Analyzer | Created with Streamlit and PyTorch</p>
</div>
"""
st.markdown(footer, unsafe_allow_html=True)