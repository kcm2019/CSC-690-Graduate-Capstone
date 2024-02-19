from flask import Flask, render_template
app = Flask(__name__)

@app.route("/")
@app.route("/home")
def home():
    return render_template('home.html')

@app.route("/chat")
def chat():
    return "Chat"

if __name__ == '__main__':
    app.run(debug=True,port=3002)