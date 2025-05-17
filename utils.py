import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import joblib
import numpy as np

import torch
import torch.nn as nn
from torchvision import models

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
    vectorizer = joblib.load("model_artifacts/vectorizer.pkl")
    rf_model = joblib.load("model_artifacts/rf_model.pkl")

    image_model = EfficientNetClassifier(output_dim=12)

    image_model.load_state_dict(torch.load("model_artifacts/image_model.pth", map_location="cpu"))
    image_model.eval()  

    return vectorizer, rf_model, image_model


def preprocess_image(image_file):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])
    image = Image.open(image_file).convert("RGB")
    return transform(image).unsqueeze(0)  

def predict_multimodal(image_model, rf_model, vectorizer, image_files, text):
    all_img_probs = []
    for file in image_files:
        image_tensor = preprocess_image(file)
        with torch.no_grad():
            logits = image_model(image_tensor)
            probs = torch.softmax(logits, dim=1).numpy()
            all_img_probs.append(probs)

    avg_img_probs = np.mean(np.vstack(all_img_probs), axis=0) 

    text_features = vectorizer.transform([text])
    text_probs = rf_model.predict_proba(text_features)[0]  

    final_probs = 0.5 * avg_img_probs + 0.5 * text_probs
    predicted_class = int(np.argmax(final_probs))
    print(predicted_class,final_probs.tolist())

    return int(predicted_class), final_probs.tolist()

def predict_text(rf_model, vectorizer, text):
    text_features = vectorizer.transform([text])
    text_probs = rf_model.predict_proba(text_features)[0]  
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
        all_probs.append(img_probs[0]) 
    
    avg_probs = np.mean(all_probs, axis=0)
    predicted_class = int(np.argmax(avg_probs))
    print(predicted_class,avg_probs.tolist())
    return predicted_class, avg_probs.tolist()
