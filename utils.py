import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import joblib
import numpy as np

import torch
import torch.nn as nn
from torchvision import models

# Define the EfficientNetClassifier class (same as in Colab)
class EfficientNetClassifier(nn.Module):
    def __init__(self, output_dim=12):
        super(EfficientNetClassifier, self).__init__()
        self.base_model = models.efficientnet_b0(pretrained=True)
        self.base_model.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(self.base_model.classifier[1].in_features, output_dim)
        )

    def forward(self, x):
        return self.base_model(x)

def load_models():
    # Load TF-IDF vectorizer and RF model
    vectorizer = joblib.load("model_artifacts/vectorizer.pkl")
    rf_model = joblib.load("model_artifacts/rf_model.pkl")

    # Initialize the model architecture
    image_model = EfficientNetClassifier(output_dim=12)

    # Load the model weights
    image_model.load_state_dict(torch.load("model_artifacts/image_model.pth", map_location="cpu"))
    image_model.eval()  # Set the model to evaluation mode

    return vectorizer, rf_model, image_model


def preprocess_image(image_file):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406], 
            std=[0.229, 0.224, 0.225]
        ),
    ])
    image = Image.open(image_file).convert("RGB")
    return transform(image).unsqueeze(0)  # Shape: (1, 3, 224, 224)

def predict(image_model, rf_model, vectorizer, image_tensor, text):
    with torch.no_grad():
        img_logits = image_model(image_tensor)
        img_probs = torch.softmax(img_logits, dim=1).numpy()

    text_features = vectorizer.transform([text])
    text_probs = rf_model.predict_proba(text_features)

    # Weighted average (late fusion)
    final_probs = 0.5 * img_probs + 0.5 * text_probs
    predicted_class = np.argmax(final_probs)

    return predicted_class, final_probs.tolist()
