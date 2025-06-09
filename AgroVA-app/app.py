from flask import Flask, redirect, session, render_template, Blueprint, request, jsonify
from auth import auth_bp
from utils import utils_bp
from chatbot.chat import chatbot_response 

app = Flask(__name__)

app.secret_key = 'supersecretkey'

# Register blueprints
app.register_blueprint(auth_bp)
app.register_blueprint(utils_bp)

@app.route('/')
def index():
    return redirect('/login')

@app.route('/dashboard')
def dashboard():
    if 'user' not in session:
        return redirect('/login')
    return render_template('dashboard.html', user=session['user'])

@app.route('/predict')
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
