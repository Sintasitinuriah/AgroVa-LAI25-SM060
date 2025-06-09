from flask import Blueprint, render_template, request, redirect, session, url_for

auth_bp = Blueprint('auth', __name__)
users = {'admin': 'admin123'}

@auth_bp.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        u, p = request.form['username'], request.form['password']
        if users.get(u) == p:
            session['user'] = u
            return redirect('/dashboard')
        return render_template('login.html', error='Login gagal!')
    return render_template('login.html')

@auth_bp.route('/logout')
def logout():
    session.pop('user', None)
    return redirect('/login')
