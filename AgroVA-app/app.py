import os
from flask import Flask, redirect, session, render_template, Blueprint, request, jsonify
from auth import auth_bp
from utils import utils_bp
from chatbot.chat import chatbot_response 
from tensorflow.keras.preprocessing.image import img_to_array
import tensorflow as tf
from PIL import Image
import numpy as np
import base64
import io
import gdown

# Suppress TensorFlow logging
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
tf.get_logger().setLevel('ERROR')

app = Flask(__name__)

# Configuration
app.secret_key = os.environ.get('SECRET_KEY', 'fallback-secret-key')
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max upload size

# Download models from Google Drive
os.makedirs('models', exist_ok=True)

def download_model_from_drive(file_id, output_path):
    if not os.path.exists(output_path):
        url = f'https://drive.google.com/uc?id={file_id}'
        print(f"Downloading model to {output_path}...")
        gdown.download(url, output_path, quiet=False)
    else:
        print(f"Model {output_path} already exists.")

# Download each model
download_model_from_drive("1fP67YM_XxCB99SWBqS_IQbJt2hAiBQzz", "models/predict_model_disease_wheat.tflite")
download_model_from_drive("1_R3Jndx3nXGWu561HitfW7yzmXUzq-6e", "models/predict_model_disease_rice.tflite")
download_model_from_drive("1Ai46UKqNkHfb7U04CUzJHyx-YeNr8V0K", "models/predict_model_disease_potato.tflite")
download_model_from_drive("1PK-1OYDZEQ8Y6n3NGr7-OjItou0EwrzY", "models/predict_model_disease_cassava.tflite")
download_model_from_drive("1tzOwkeyD1HdAo1C9LXO8JlhilPtngKIV", "models/converted_model.tflite")

# Model Loading Function
def load_tflite_model(model_path):
    try:
        interpreter = tf.lite.Interpreter(model_path=model_path)
        interpreter.allocate_tensors()
        return interpreter
    except Exception as e:
        print(f"Error loading model {model_path}: {str(e)}")
        raise

# Load Main Model
try:
    interpreter = load_tflite_model("models/converted_model.tflite")
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
except Exception as e:
    print(f"Failed to load main model: {str(e)}")
    interpreter = None

label_map = [
    'gandum_Healthy', 'gandum_septoria', 'gandum_stripe_rust',
    'kentang_Early_Blight', 'kentang_Healthy', 'kentang_Late_Blight',
    'padi_bacterial_leaf_blight', 'padi_bacterial_leaf_streak', 'padi_bacterial_panicle_blight',
    'padi_blast', 'padi_brown_spot', 'padi_dead_heart', 'padi_downy_mildew',
    'padi_hispa', 'padi_normal', 'padi_tungro',
    'singkong_Cassava_Bacterial_Blight_(CBB)', 'singkong_Cassava_Brown_Streak_Disease_(CBSD)',
    'singkong_Cassava_Green_Mottle_(CGM)', 'singkong_Cassava_Mosaic_Disease_(CMD)', 'singkong_Healthy',
]

model_map = {
    'padi': {
        'model': tf.lite.Interpreter(model_path='models/predict_model_disease_rice.tflite'),
        'labels': ['Bacterialblight', 'Blast', 'Brownspot', 'Tungro']
    },
    'kentang': {
        'model': tf.lite.Interpreter(model_path='models/predict_model_disease_potato.tflite'),
        'labels': ['Early_Blight', 'Healthy', 'Late_Blight']
    },
    'gandum': {
        'model': tf.lite.Interpreter(model_path='models/predict_model_disease_wheat.tflite'),
        'labels': [
            'aphid', 'black_rust', 'blast', 'brown_rust', 'common_root_rot',
            'fusarium_head_blight', 'healthy', 'leaf_blight', 'mildew', 'mite',
            'septoria', 'smut', 'stem_fly', 'tan_spot', 'yellow_rust'
        ]
    },
    'singkong': {
        'model': tf.lite.Interpreter(model_path='models/predict_model_disease_cassava.tflite'),
        'labels': [
            'Cassava__bacterial_blight', 'Cassava__brown_streak_disease',
            'Cassava__green_mottle', 'Cassava__healthy', 'Cassava__mosaic_disease'
        ]
    }
}

for entry in model_map.values():
    entry['model'].allocate_tensors()

# Register blueprints
app.register_blueprint(auth_bp)
app.register_blueprint(utils_bp)

def preprocess_image(image, target_size=(224, 224)):
    if image.mode != 'RGB':
        image = image.convert('RGB')
    image = image.resize(target_size)
    img_array = img_to_array(image) / 255.0
    return np.expand_dims(img_array, axis=0)

def predict_image(image):
    try:
        processed_image = preprocess_image(image)
        interpreter.set_tensor(input_details[0]['index'], processed_image)
        interpreter.invoke()
        return interpreter.get_tensor(output_details[0]['index'])[0]
    except Exception as e:
        print(f"Prediction error: {str(e)}")
        raise

def predict_with_tflite(interpreter, image_array):
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    input_data = np.expand_dims(image_array.astype(np.float32), axis=0)
    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()

    output_data = interpreter.get_tensor(output_details[0]['index'])
    return output_data

@app.route('/')
def index():
    return redirect('/login')

@app.route('/predict-main', methods=['POST'])
def predict():
    result = None
    image = None

    if 'imageData' in request.form and request.form['imageData']:
        image_data = request.form['imageData'].split(',')[1]
        image = Image.open(io.BytesIO(base64.b64decode(image_data)))
    elif 'image' in request.files:
        image = Image.open(request.files['image'].stream)

    if image:
        prediction = predict_image(image)
        label_full = label_map[np.argmax(prediction)]

        # Format output
        if '_' in label_full:
            plant, disease = label_full.split('_', 1)
            result = f"{plant.capitalize()} - {disease.replace('_', ' ')}"
        else:
            result = label_full

    return render_template('weather.html', result=result)

@app.route('/predict-plant', methods=['POST'])
def predict_model():
    plant = request.form.get('plant')
    result_modal = None
    image = None

    if not plant or plant not in model_map:
        return render_template('weather.html', result_modal='Tanaman tidak dikenali.', plant=plant)

    if 'imageData' in request.form and request.form['imageData']:
        image_data = request.form['imageData'].split(',')[1]
        image = Image.open(io.BytesIO(base64.b64decode(image_data))).convert('RGB')
    elif 'image' in request.files:
        image = Image.open(request.files['image'].stream).convert('RGB')

    if image:
        model_info = model_map[plant]
        interpreter = model_info['model']
        label_map = model_info['labels']

        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()

        input_shape = input_details[0]['shape'][1:3]
        img = image.resize(input_shape)
        img_array = np.array(img).astype(np.float32) / 255.0
        input_data = np.expand_dims(img_array, axis=0)

        interpreter.set_tensor(input_details[0]['index'], input_data)
        interpreter.invoke()
        output_data = interpreter.get_tensor(output_details[0]['index'])[0]

        predicted_index = np.argmax(output_data)
        predicted_label = label_map[predicted_index]

        result_modal = predicted_label.replace('_', ' ').replace('Cassava__', '').capitalize()

    return render_template('weather.html', result_modal=result_modal, plant=plant)

@app.route('/dashboard')
def dashboard():
    if 'user' not in session:
        return redirect('/login')
    return render_template('dashboard.html', user=session['user'])

@app.route('/predict-main')
def predict_weather():
    if 'user' not in session:
        return redirect('/login')
    return render_template('weather.html', user=session['user'])

@app.route('/chat')
def chat():
    if 'user' not in session:
        return redirect('/login')
    return render_template('chatbot.html', user=session['user'])

@app.route("/chatbot", methods=["POST"])
def get_bot_response():
    data = request.get_json()
    user_input = data.get("message")
    intent, response = chatbot_response(user_input)
    return jsonify({"intent": intent, "response": response})

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
