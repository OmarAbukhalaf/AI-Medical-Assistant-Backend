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

def predict_multimodal(image_model, rf_model, vectorizer, image_files, text):
    # Predict and average over all image files
    all_img_probs = []
    for file in image_files:
        image_tensor = preprocess_image(file)
        with torch.no_grad():
            logits = image_model(image_tensor)
            probs = torch.softmax(logits, dim=1).numpy()
            all_img_probs.append(probs)

    avg_img_probs = np.mean(np.vstack(all_img_probs), axis=0)  # shape (num_classes,)

    # Text probabilities
    text_features = vectorizer.transform([text])
    text_probs = rf_model.predict_proba(text_features)[0]  # shape (num_classes,)

    # Weighted late fusion
    final_probs = 0.5 * avg_img_probs + 0.5 * text_probs
    predicted_class = int(np.argmax(final_probs))
    print(predicted_class,final_probs.tolist())

    return int(predicted_class), final_probs.tolist()

def predict_text(rf_model, vectorizer, text):
    text_features = vectorizer.transform([text])
    text_probs = rf_model.predict_proba(text_features)[0]  # shape: (num_classes,)
    predicted_class = int(np.argmax(text_probs))
    print(predicted_class,text_probs.tolist())
    return predicted_class, text_probs.tolist()

def predict_image(image_model, image_files):
    all_probs = []
    for image_file in image_files:
        image_tensor = preprocess_image(image_file)
        with torch.no_grad():
            img_logits = image_model(image_tensor)
            img_probs = torch.softmax(img_logits, dim=1).detach().cpu().numpy()
        all_probs.append(img_probs[0])  # assuming shape (1, num_classes)
    
    avg_probs = np.mean(all_probs, axis=0)
    predicted_class = int(np.argmax(avg_probs))
    print(predicted_class,avg_probs.tolist())
    return predicted_class, avg_probs.tolist()