from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import numpy as np

try:
    from tensorflow.keras.models import load_model 
    from tensorflow.keras.preprocessing import image 
    from PIL import Image
    TENSORFLOW_AVAILABLE = True
except ImportError:
    print("Warning: TensorFlow or Pillow not found. Running in mock mode.")
    TENSORFLOW_AVAILABLE = False
    Image = None

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

model = None
if TENSORFLOW_AVAILABLE:
    model_path = 'WheatDiseaseDetection.h5'
    if os.path.exists(model_path):
        try:
            model = load_model(model_path)
            print("Model loaded successfully.")
        except Exception as e:
            print(f"Error loading model: {e}")
    else:
        print(f"Error: Model file '{model_path}' not found.")

def prepare_image(img):
    img = img.resize((255, 255))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0
    return img_array

@app.route('/predict', methods=['POST'])
def predict():
    if not TENSORFLOW_AVAILABLE or not model:
        # Mock Response for verification
        return jsonify({
            'prediction': 'Mock - Healthy',
            'cause': 'Mock - No disease',
            'pesticide': 'Mock - No pesticide needed',
            'success': True
        })

    if 'image' not in request.files:
        return jsonify({'error': 'No image file provided'}), 400
    
    file = request.files['image']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    if file:
        try:
            img = Image.open(file.stream).convert('RGB')
            img_array = prepare_image(img)
            predictions = model.predict(img_array)
            predicted_class = np.argmax(predictions, axis=1)[0]
            
            labels = {
                0: 'Aphid', 1: 'Black Rust', 2: 'Blast', 3: 'Brown Rust', 4: 'Common Root Rot',
                5: 'Fusarium Head Blight', 6: 'Healthy', 7: 'Leaf Blight', 8: 'Mildew', 9: 'Mite',
                10: 'Septoria', 11: 'Smut', 12: 'Stem fly', 13: 'Tan spot', 14: 'Yellow Rust'
            }
            
            pesticide_map = {
                0: 'Imidacloprid, Thiamethoxam, or Pyrethroids',
                1: 'Propiconazole, Tebuconazole, or Mancozeb',
                2: 'Tricyclazole or Isoprothiolane',
                3: 'Propiconazole, Mancozeb, or Tebuconazole',
                4: 'Thiram, Carboxin, or Mancozeb',
                5: 'Tebuconazole, Metconazole, or Prothioconazole',
                6: 'No pesticide needed',
                7: 'Chlorothalonil or Propiconazole',
                8: 'Sulphur, Triadimefon, or Fenarimol',
                9: 'Abamectin, Bifenthrin, or Spiromesifen',
                10: 'Chlorothalonil, Tebuconazole, or Epoxiconazole',
                11: 'Carboxin, Thiram, or Carbendazim',
                12: 'Imidacloprid or Thiamethoxam',
                13: 'Propiconazole, Tebuconazole, or Mancozeb',
                14: 'Tebuconazole, Propiconazole, or Azoxystrobin'
            }
            
            causes = {
                0: 'Aphid infestation',
                1: 'Fungal infection by Puccinia graminis',
                2: 'Fungal infection by Magnaporthe oryzae',
                3: 'Fungal infection by Puccinia triticina',
                4: 'Fungal infection by Bipolaris sorokiniana',
                5: 'Fungal infection by Fusarium species',
                6: 'No disease (healthy plant)',
                7: 'Fungal infection by Helminthosporium species',
                8: 'Fungal infection by Erysiphe graminis',
                9: 'Mite infestation (e.g., Wheat Curl Mite)',
                10: 'Fungal infection by Zymoseptoria tritici',
                11: 'Fungal infection by Ustilago species',
                12: 'Infestation by Wheat Stem Fly larvae',
                13: 'Fungal infection by Pyrenophora tritici-repentis',
                14: 'Fungal infection by Puccinia striiformis',
            }
            
            return jsonify({
                'prediction': labels.get(predicted_class, "Unknown"),
                'cause': causes.get(predicted_class, "Unknown"),
                'pesticide': pesticide_map.get(predicted_class, "Consult an expert"),
                'success': True
            })

        except Exception as e:
            return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)
