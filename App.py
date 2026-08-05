import streamlit as st
import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import cv2
import numpy as np
from facenet_pytorch import MTCNN

from model import get_model

# Streamlit Page Setup
st.set_page_config(page_title="Autism Classification System", layout="centered")

st.title("Autism Classification System")
st.caption("Hybrid Deep Learning using Facial Images (ViT + SE + CBAM)")

# Initialize MTCNN for face detection and validation
@st.cache_resource
def load_mtcnn():
    return MTCNN(keep_all=False, select_largest=True, post_process=False)

# Load the trained PyTorch Hybrid Model
@st.cache_resource
def load_trained_model():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = get_model(num_classes=2, device=device)
    
    # Load model weights if available (or evaluate model in eval mode)
    try:
        model.load_state_dict(torch.load("model_weights.pth", map_location=device))
    except FileNotFoundError:
        st.warning("Pre-trained weights file 'model_weights.pth' not found. Running with initial weights.")
        
    model.eval()
    return model, device

mtcnn = load_mtcnn()
model, device = load_trained_model()

# Image Preprocessing Transforms (Resizing & Normalization)
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

def check_blur(cv_image, threshold=100.0):
    """Calculates Laplacian variance to detect blurred image inputs."""
    gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
    variance = cv2.Laplacian(gray, cv2.CV_64F).var()
    return variance < threshold

# Sidebar Controls
st.sidebar.header("Controls")
st.sidebar.markdown("1. Upload a clear facial image\n2. Face validation\n3. Classification result")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Read Image File
    pil_image = Image.open(uploaded_file).convert('RGB')
    cv_image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)

    st.image(pil_image, caption="Uploaded Image", use_column_width=True)

    st.subheader("Analysis Results")

    # Step 1: Quality Check for Blur
    if check_blur(cv_image, threshold=50.0):
        st.error("Result: Blurred Image")
        st.info("Please upload a clear facial image.")
    else:
        # Step 2: Human Face Validation using MTCNN
        boxes, _ = mtcnn.detect(pil_image)
        if boxes is None:
            st.error("Result: Invalid Input")
            st.info("Non-facial image detected. Please upload a valid human facial photo.")
        else:
            # Step 3: Model Inference & Classification
            img_tensor = transform(pil_image).unsqueeze(0).to(device)

            with torch.no_grad():
                output = model(img_tensor)
                probs = torch.softmax(output, dim=1)
                confidence, pred = torch.max(probs, dim=1)

            class_names = ["Non-Autism", "Autism"]
            result_class = class_names[pred.item()]
            score_percent = confidence.item() * 100

            # Render Classification Results
            if result_class == "Autism":
                st.error(f"Prediction: **{result_class}**")
            else:
                st.success(f"Prediction: **{result_class}**")

            st.metric(label="Confidence Score", value=f"{score_percent:.2f}%")
            st.progress(float(confidence.item()))
