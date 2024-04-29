from flask import Flask, render_template, request, jsonify
import random

app = Flask(__name__)

numbers_history = []

def add_numbers(num1, num2):
    result = num1 + num2
    numbers_history.append((num1, num2, result))
    return result

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        num1, num2 = map(int, request.form['numbers'].split())
        result = add_numbers(num1, num2)
        return jsonify({'result': result, 'history': numbers_history})
    return render_template('math.html', history=numbers_history)

if __name__ == '__main__':
    app.run(debug=True)
