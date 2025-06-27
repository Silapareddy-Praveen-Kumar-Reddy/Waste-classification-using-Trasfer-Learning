from flask import Flask, render_template, request, redirect, url_for, flash, send_from_directory
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.applications.vgg16 import preprocess_input
from werkzeug.utils import secure_filename
from PIL import Image
import numpy as np
import os

app = Flask(__name__)
app.secret_key = 'waste-management-secret-key-2024'

UPLOAD_FOLDER = 'static/uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'bmp'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

model = load_model('vgg16.h5')

WASTE_CLASSES = ['Biodegradable Images', 'Recyclable Images', 'Trash Images']

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def preprocess_image(image_path, target_size=(224, 224)):
    try:
        img = Image.open(image_path).convert("RGB")
        img = img.resize(target_size)
        img_array = img_to_array(img)
        img_array = preprocess_input(img_array)
        return np.expand_dims(img_array, axis=0)
    except Exception as e:
        print(f"Image preprocessing error: {e}")
        return None

def get_prediction_details(prediction_array, classes):
    predicted_class_idx = int(np.argmax(prediction_array))

    if predicted_class_idx >= len(classes):
        predicted_class = "Unknown"
    else:
        predicted_class = classes[predicted_class_idx]

    confidence = float(prediction_array[predicted_class_idx])
    all_probabilities = {
        classes[i] if i < len(classes) else f"Class_{i}": float(prediction_array[i])
        for i in range(len(prediction_array))
    }

    return {
        'predicted_class': predicted_class,
        'confidence': confidence,
        'confidence_percentage': confidence * 100,
        'all_probabilities': all_probabilities
    }

def get_recycling_instructions(waste_class):
    instructions = {
        'Biodegradable Images': {
            'disposal': 'Use compost bin or biodegradable waste service.',
            'tips': 'Can be composted at home or through city programs.',
            'environmental_impact': 'Low - breaks down naturally.'
        },
        'Recyclable Images': {
            'disposal': 'Use local recycling bin or station.',
            'tips': 'Clean before disposal and sort properly.',
            'environmental_impact': 'Medium - recyclable but must be cleaned.'
        },
        'Trash Images': {
            'disposal': 'Place in general waste.',
            'tips': 'Avoid if recyclable options exist.',
            'environmental_impact': 'High - contributes to landfill.'
        }
    }
    return instructions.get(waste_class, {
        'disposal': 'Check local waste guidelines.',
        'tips': 'Sort based on material.',
        'environmental_impact': 'Variable'
    })
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'GET':
        return render_template('predict.html')

    if 'file' not in request.files:
        flash('No file uploaded', 'error')
        return redirect(url_for('predict'))

    file = request.files['file']
    if file.filename == '' or not allowed_file(file.filename):
        flash('Invalid or no file selected.', 'error')
        return redirect(url_for('predict'))

    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)

    processed_image = preprocess_image(filepath)
    if processed_image is None:
        flash('Error processing image.', 'error')
        return redirect(url_for('predict'))

    prediction = model.predict(processed_image, verbose=0)
    print("Prediction shape:", prediction)
    print("WASTE_CLASSES:", WASTE_CLASSES)

    prediction_details = get_prediction_details(prediction[0], WASTE_CLASSES)
    recycling_info = get_recycling_instructions(prediction_details['predicted_class'])

    result_data = {
        'filename': filename,
        'filepath': url_for('static', filename=f'uploads/{filename}'),
        'prediction': prediction_details,
        'recycling_info': recycling_info
    }

    return render_template('portfolio.html', result=result_data)

@app.route('/blog')
def blog():
    return render_template('blog.html')
@app.route('/contact')
def contact():
    return render_template('contact.html')

if __name__ == '__main__':
    app.run(debug=True, port=2222)
