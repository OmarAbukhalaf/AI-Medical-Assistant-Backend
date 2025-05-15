from flask import Flask, request, jsonify, render_template
from flask_cors import CORS  
import torch
from utils import load_models, preprocess_image, predict_multimodal,predict_image,predict_text
import os
import numpy as np
app = Flask(__name__)
CORS(app)
# Load models on startup
vectorizer, rf_model, image_model = load_models()

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict_route():
    text_input = request.form.get('text', '').strip()
    image_files = request.files.getlist('images')

    try:
        if text_input and image_files and image_files[0].filename != '':
            # Both text and images
            predicted_class, probs = predict_multimodal(image_model, rf_model, vectorizer, image_files, text_input)
        elif text_input and (not image_files or image_files[0].filename == ''):
            # Text only
            predicted_class, probs = predict_text(rf_model, vectorizer, text_input)

        elif image_files and image_files[0].filename != '':
            # Images only
            predicted_class, probs = predict_image(image_model, image_files)

        else:
            return jsonify({"error": "No valid input provided (text or images required)"}), 400

        return jsonify({
            "predicted_class": predicted_class,
            "class_probabilities": probs
        })

    except Exception as e:
        import traceback
        print(traceback.format_exc())
        return jsonify({"error": str(e)}), 500



if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)