from flask import Flask, request, render_template, redirect
from tensorflow.keras.models import load_model 
from tensorflow.keras.preprocessing import image 
from PIL import Image
import numpy as np

app = Flask(__name__)
model = load_model('WheatDiseaseDetection.h5')

def prepare_image(img):
    img = img.resize((255, 255))  # Update to match your target size
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0  # Rescaling
    return img_array

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return redirect(request.url)
    
    file = request.files['image']
    if file.filename == '':
        return redirect(request.url)
    
    if file:
        # Convert FileStorage to PIL Image
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
        0: 'Aphid infestation',                            # Aphid
        1: 'Fungal infection by Puccinia graminis',        # Black Rust
        2: 'Fungal infection by Magnaporthe oryzae',       # Blast
        3: 'Fungal infection by Puccinia triticina',       # Brown Rust
        4: 'Fungal infection by Bipolaris sorokiniana',    # Common Root Rot
        5: 'Fungal infection by Fusarium species',         # Fusarium Head Blight
        6: 'No disease (healthy plant)',                   # Healthy
        7: 'Fungal infection by Helminthosporium species', # Leaf Blight
        8: 'Fungal infection by Erysiphe graminis',        # Mildew (Powdery Mildew)
        9: 'Mite infestation (e.g., Wheat Curl Mite)',     # Mite
        10: 'Fungal infection by Zymoseptoria tritici',    # Septoria
        11: 'Fungal infection by Ustilago species',        # Smut
        12: 'Infestation by Wheat Stem Fly larvae',        # Stem fly
        13: 'Fungal infection by Pyrenophora tritici-repentis', # Tan spot
        14: 'Fungal infection by Puccinia striiformis',    # Yellow Rust
        }
        cause=causes[predicted_class]

        prediction = labels[predicted_class]
        pesticide = pesticide_map[predicted_class]
        return render_template('index.html', prediction=prediction, pesticide=pesticide, cause=cause, image_url=file.filename)

if __name__ == '__main__':
    app.run(debug=True)
