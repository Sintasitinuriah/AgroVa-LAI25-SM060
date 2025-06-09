from flask import Blueprint, render_template, request, session

utils_bp = Blueprint('utils', __name__)

@utils_bp.route('/weather', methods=['GET', 'POST'])
def weather():
    if 'user' not in session:
        return "Unauthorized", 401
    result = None
    if request.method == 'POST':
        temp = float(request.form['temperature'])
        humid = float(request.form['humidity'])
        result = "Hujan" if humid > 80 else "Cerah"
    return render_template('weather.html', result=result)
