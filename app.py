import streamlit as st
import torch
import torchvision.transforms as transforms
from PIL import Image
import torchvision.models as models
import torch.nn as nn

# Set Streamlit Page Config
st.set_page_config(page_title="Deepfake Image Detector", layout="centered")

# Load Trained Models
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load ResNet18
resnet18 = models.resnet18(pretrained=False)
resnet18.fc = nn.Linear(resnet18.fc.in_features, 2)
resnet18.load_state_dict(torch.load("resnet18_deepfake.pth", map_location=device))
resnet18.to(device)
resnet18.eval()

# Load EfficientNet-B0
efficientnet = models.efficientnet_b0(pretrained=False)
efficientnet.classifier[1] = nn.Linear(efficientnet.classifier[1].in_features, 2)
efficientnet.load_state_dict(torch.load("efficientnet_b0_deepfake.pth", map_location=device))
efficientnet.to(device)
efficientnet.eval()

# Define Image Preprocessing
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Streamlit UI
st.markdown("<h1 style='text-align: center; color: #4A90E2;'> Deepfake Image Detector</h1>", unsafe_allow_html=True)
st.write("<p style='text-align: center; font-size: 18px;'>Upload an image and select a model to determine if it's **Real or Fake**.</p>", unsafe_allow_html=True)

# Model Selection
st.sidebar.title(" Choose a Model")
model_choice = st.sidebar.radio("Select Model:", ["ResNet18", "EfficientNet-B0"])

# Upload Image
uploaded_file = st.file_uploader(" Upload an image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    
    # Show Uploaded Image
    st.image(image, caption=" Uploaded Image", use_column_width=True)

    # Preprocess Image
    img_tensor = transform(image).unsqueeze(0).to(device)

    # Select Model
    if model_choice == "ResNet18":
        model = resnet18
    else:
        model = efficientnet

    # Predict
    with torch.no_grad():
        output = model(img_tensor)
        _, prediction = torch.max(output, 1)
    
    label = "🟥 FAKE" if prediction.item() == 0 else "🟩 REAL"

    # Display Result with Styled Text
    st.markdown(f"<h2 style='text-align: center; color: #D32F2F;'>{model_choice} Prediction:</h2>", unsafe_allow_html=True)
    st.markdown(f"<h1 style='text-align: center; color: #2E7D32;'>{label}</h1>", unsafe_allow_html=True)

    # Get softmax probabilities
    softmax = torch.nn.Softmax(dim=1)
    probabilities = softmax(output)

    # Extract confidence score
    confidence = probabilities[0][prediction.item()].item() * 100

    #   Display result with confidence
    st.markdown(f"<h1 style='text-align: center;'>{label} ({confidence:.2f}%)</h1>", unsafe_allow_html=True)


    # Model Summary
    st.sidebar.subheader(" Model Summary")
    if model_choice == "ResNet18":
        st.sidebar.write(" **ResNet18:**\n- **Fast & Lightweight**\n- **Good Accuracy**\n- **Better for Smaller Datasets**")
    else:
        st.sidebar.write(" **EfficientNet-B0:**\n- **High Accuracy**\n- **More Computationally Expensive**\n- **Better for Complex Images**")

# Footer
st.markdown("<br><p style='text-align: center; font-size: 14px; color: gray;'> Built with PyTorch & Streamlit | Created by Arjun</p>", unsafe_allow_html=True)
