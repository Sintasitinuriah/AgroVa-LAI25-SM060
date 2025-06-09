from flask import Flask, redirect, session, render_template, Blueprint, request, jsonify
from auth import auth_bp
from utils import utils_bp
from chatbot.chat import chatbot_response 
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
import tensorflow as tf
from PIL import Image
import numpy as np
import os
import base64
import io

app = Flask(__name__)

app.secret_key = 'supersecretkey'
interpreter = tf.lite.Interpreter(model_path="C:/Users/Sinta/Documents/Larskar AI/Capstone/klasifikasi gambar/model/tflite_model/converted_model.tflite")
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

label_map = [
    'gandum_Healthy',
    'gandum_septoria',
    'gandum_stripe_rust',
    'kentang_Early_Blight',
    'kentang_Healthy',
    'kentang_Late_Blight',
    'padi_bacterial_leaf_blight',
    'padi_bacterial_leaf_streak',
    'padi_bacterial_panicle_blight',
    'padi_blast',
    'padi_brown_spot',
    'padi_dead_heart',
    'padi_downy_mildew',
    'padi_hispa',
    'padi_normal',
    'padi_tungro',
    'singkong_Cassava_Bacterial_Blight_(CBB)',
    'singkong_Cassava_Brown_Streak_Disease_(CBSD)',
    'singkong_Cassava_Green_Mottle_(CGM)',
    'singkong_Cassava_Mosaic_Disease_(CMD)',
    'singkong_Healthy',
]


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
    processed_image = preprocess_image(image)
    interpreter.set_tensor(input_details[0]['index'], processed_image)
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]['index'])
    return output_data[0]

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
    print("Received:", data)
    print("Request method:", request.method)
    print("Request headers:", request.headers)
    print("Request JSON:", request.get_json())
    user_input = data.get("message")
    intent, response = chatbot_response(user_input)
    return jsonify({"intent": intent, "response": response})

if __name__ == '__main__':
    app.run(debug=True)
