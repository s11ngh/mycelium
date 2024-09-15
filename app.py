# app.py
from flask import Flask, jsonify

app = Flask(__name__)

# Define a simple route
@app.route('/')
def home():
    return jsonify(message="Test Local Host")

# Define a route for health check
@app.route('/health')
def health():
    return jsonify(status="OK")

# You can add other routes to handle PyGrid or other tasks
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
