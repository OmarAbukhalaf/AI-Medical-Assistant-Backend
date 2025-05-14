from flask import Flask, request, jsonify, render_template
from flask_cors import CORS  
from utils import load_models, preprocess_image, predict

app = Flask(__name__)
CORS(app)
# Load models on startup
vectorizer, rf_model, image_model = load_models()

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict_route():
    if 'text' not in request.form or 'images' not in request.files:
        return jsonify({"error": "Missing images or text input"}), 400

    text_input = request.form['text']
    image_files = request.files.getlist('images')

    try:
        all_probs = []
        all_predictions = []

        for image_file in image_files:
            image_tensor = preprocess_image(image_file)
            predicted_class, probs = predict(image_model, rf_model, vectorizer, image_tensor, text_input)
            all_predictions.append(predicted_class)
            all_probs.append(probs[0])  # assuming probs is [[...]], take first element

        # Example logic: average probabilities across all images
        import numpy as np
        avg_probs = np.mean(all_probs, axis=0)
        final_predicted_class = int(np.argmax(avg_probs))

        return jsonify({
            "predicted_class": final_predicted_class,
            "class_probabilities": avg_probs.tolist()
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == '__main__':
    app.run(debug=True)
