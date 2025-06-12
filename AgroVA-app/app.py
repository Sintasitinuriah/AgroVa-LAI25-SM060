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
download_model_from_drive("1g60wbq4iVVsULpN45L_IvkGGipKi7KOS", "models/model_predict_cuaca.tflite")

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

interpreter_cuaca = None
try:
    model_path = 'models/model_predict_cuaca.tflite'
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"File model tidak ditemukan di: {model_path}")
    
    interpreter_cuaca = tf.lite.Interpreter(model_path=model_path)
    interpreter_cuaca.allocate_tensors()  # Ini juga penting!
    
    input_details_cuaca = interpreter_cuaca.get_input_details()
    output_details_cuaca = interpreter_cuaca.get_output_details()

    print("[INFO] Model cuaca berhasil dimuat.")

except Exception as e:
    print("[ERROR] Gagal memuat model cuaca:", e)
    interpreter_cuaca = None

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

def predict_image_model(interpreter, image_array):
    """
    Melakukan prediksi menggunakan model berbasis gambar (misal CNN).
    image_array harus berukuran (height, width, 3) dan sudah dinormalisasi.
    """
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    input_data = np.expand_dims(image_array.astype(np.float32), axis=0)  # (1, H, W, 3)
    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()

    output_data = interpreter.get_tensor(output_details[0]['index'])
    return output_data


def predict_timeseries_model(interpreter, timeseries_array):
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    expected_shape = input_details[0]['shape']

    print("Expected model input shape:", expected_shape)
    print("Data shape:", np.array(timeseries_array).shape)

    try:
        data_np = np.array(timeseries_array, dtype=np.float32)

        if list(expected_shape) == [1, 7, 12] and data_np.shape == (7, 12):
            input_array = data_np.reshape(1, 7, 12)

        elif list(expected_shape) == [1, 84] and data_np.size == 84:
            input_array = data_np.reshape(1, 84)

        elif list(expected_shape) == [1, 7, 1]:
            fitur_tunggal = [row[2] for row in timeseries_array]
            input_array = np.array(fitur_tunggal, dtype=np.float32).reshape(1, 7, 1)

        else:
            raise ValueError(f"Shape input model tidak cocok dengan data: {data_np.shape}")

        interpreter.set_tensor(input_details[0]['index'], input_array)
        interpreter.invoke()
        output_data = interpreter.get_tensor(output_details[0]['index'])
        return output_data

    except Exception as e:
        print(f"Error predicting: {e}")
        return None

def predict_with_tflite(interpreter, data):
    """
    Fungsi general untuk prediksi TFLite model.
    - Jika input shape (1, 224, 224, 3) → diasumsikan model gambar
    - Jika input shape (1, 7, 1) → diasumsikan model time series
    """
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    expected_shape = input_details[0]['shape']

    # Model gambar
    if len(expected_shape) == 4 and list(expected_shape[1:]) == [224, 224, 3]:
        input_data = np.expand_dims(data.astype(np.float32), axis=0)

    # Model time series
    elif len(expected_shape) == 3 and list(expected_shape[1:]) == [7, 1]:
        print("Input shape:", interpreter.get_input_details()[0]['shape'])
        input_data = np.array(data, dtype=np.float32).reshape(1, 7, 1)

    else:
        raise ValueError(f"Input shape {expected_shape} tidak dikenali. "
                         f"Harus gambar (224x224x3) atau time series (7x1)")

    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()
    return interpreter.get_tensor(output_details[0]['index'])


def predict_with_models(interpreter, data_7_hari):
    results = []

    try:
        cuaca_map = {0: "Cerah", 1: "Hujan", 2: "Berawan", 3: "Mendung"}

        # Ambil info input dan output
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()

        print("[DEBUG] Input shape model:", input_details[0]['shape'])
        print("[DEBUG] Output shape model:", output_details[0]['shape'])

        data = data_7_hari.copy()  # Pastikan tidak ubah data asli

        for i in range(7):
            # Ambil fitur suhu_min dari tiap hari (fitur ke-3, index 2)
            fitur_tunggal = [row[2] for row in data]
            input_array = np.array(fitur_tunggal, dtype=np.float32).reshape(1, 7, 1)

            # Validasi bentuk input
            expected_shape = tuple(input_details[0]['shape'])
            if input_array.shape != expected_shape:
                raise ValueError(f"Bentuk input tidak cocok: {input_array.shape}, seharusnya: {expected_shape}")

            # Prediksi
            interpreter.set_tensor(input_details[0]['index'], input_array)
            interpreter.invoke()
            output = interpreter.get_tensor(output_details[0]['index'])[0]

            print(f"[DEBUG] Output prediksi hari ke-{i+1}:", output)

            if len(output) < 2:
                raise ValueError("Output model tidak lengkap. Harus [suhu, prob_cerah, prob_hujan, ...]")

            pred_suhu = output[0]
            pred_cuaca_probs = output[1:]

            label_cuaca = np.argmax(pred_cuaca_probs)
            cuaca = cuaca_map.get(label_cuaca, "Tidak diketahui")

            results.append({
                "day": i + 1,
                "temperature": round(pred_suhu, 1),
                "wheater": cuaca
            })

            # Update data untuk hari berikutnya
            fitur_baru = data[-1][:]
            fitur_baru[2] = pred_suhu  # Update suhu_min dengan prediksi
            data.pop(0)
            data.append(fitur_baru)

    except Exception as e:
        print(f"[ERROR] Gagal prediksi: {e}")

        # Fallback prediksi statis jika model gagal
        results = [
            {"day": i + 1, "temperature": 30.0 + (i % 3), "wheater": "Cerah"}
            for i in range(7)
        ]

    return results

@app.route('/')
def index():
    if 'user' not in session:
        return redirect('/login')
    return render_template('dashboard.html', user=session['user'])


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
    data_7_hari = [
        [22, 30, 26, 70, 5, 6, 3, 180, 3, -6.2, 106.8, 8],
        [23, 31, 27, 65, 4, 7, 2, 190, 3, -6.2, 106.8, 8],
        [22, 29, 25, 72, 6, 5, 4, 170, 3, -6.2, 106.8, 7],
        [21, 28, 24, 75, 10, 4, 3, 160, 2, -6.2, 106.8, 7],
        [22, 30, 26, 68, 2, 8, 3, 180, 4, -6.2, 106.8, 8],
        [24, 32, 28, 60, 0, 9, 2, 190, 3, -6.2, 106.8, 8],
        [25, 33, 29, 58, 0, 9, 2, 200, 3, -6.2, 106.8, 8]
    ]

    if interpreter_cuaca is None:
            print("[WARNING] Interpreter cuaca tidak tersedia. Menampilkan prediksi statis.")
            prediksi = [
                {"day": i + 1, "temperature": 30.0 + (i % 2), "wheater": "Cerah"}
                for i in range(7)
            ]
    else:
        prediksi = predict_with_models(interpreter_cuaca, data_7_hari)

    return render_template('dashboard.html', user=session['user'], prediksi=prediksi)
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
