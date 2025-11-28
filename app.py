import re
import pickle
from flask import Flask, request, render_template, jsonify
import os

# ----------------------------
# If your pipeline used a custom function named `clean_text`
# it must be defined at module level BEFORE we unpickle the model.
# This implementation is robust: handles a single string or an iterable.
# ----------------------------
def clean_text(x):
    """
    Clean text helper used by the original training pipeline.
    Accepts either:
      - a single string -> returns cleaned string
      - an iterable (list/array/pandas.Series) -> returns list of cleaned strings
    """
    def _clean(s):
        if s is None:
            return ""
        s = str(s).lower()
        # remove urls
        s = re.sub(r'http\S+', ' ', s)
        # remove non alphanumeric characters (keep spaces)
        s = re.sub(r'[^a-z0-9\s]', ' ', s)
        # collapse multiple spaces
        s = re.sub(r'\s+', ' ', s).strip()
        return s

    # If input is a single string, return cleaned string
    if isinstance(x, str):
        return _clean(x)
    # If input is bytes
    if isinstance(x, (bytes, bytearray)):
        try:
            return _clean(x.decode('utf-8', errors='ignore'))
        except:
            return _clean(str(x))
    # Else assume iterable and return list
    try:
        return [_clean(item) for item in x]
    except TypeError:
        # fallback: convert to str and clean
        return _clean(str(x))


# Initialize the Flask application
app = Flask(__name__, template_folder='templates')

# Load the pre-trained model (make sure this file exists next to app.py)
model = None
model_path = 'emailspam_model.pkl'

try:
    if not os.path.exists(model_path):
        print(f"Error: Model file '{model_path}' not found. Please ensure it exists in the same folder as app.py.")
        model = None
    else:
        # NOTE: clean_text must exist before pickle.load if the pipeline references it
        with open(model_path, 'rb') as file:
            model = pickle.load(file)
        print("Model loaded successfully.")
except Exception as e:
    # Provide a helpful message so user can debug
    print(f"Error loading model: {e}")
    model = None

@app.route('/')
def home():
    return render_template('index.html', prediction_text="")

@app.route('/predict_web', methods=['POST'])
def predict_web():
    if model is None:
        return render_template('index.html', prediction_text="Model not loaded. Cannot make predictions.")
    email_text = request.form.get('email_text', '')
    # model expects a list-like input
    try:
        prediction = model.predict([email_text])[0]
    except Exception as e:
        # return an error message to the UI so you can debug
        return render_template('index.html', prediction_text=f"Prediction error: {e}")
    return render_template('index.html', prediction_text=f'The email is: {str(prediction).upper()}')

@app.route('/predict', methods=['POST'])
def predict_api():
    if model is None:
        return jsonify({'error': 'Model not loaded. Cannot make predictions.'})
    data = request.get_json(force=True)
    email = data.get('email', '')
    try:
        prediction = model.predict([email])[0]
    except Exception as e:
        return jsonify({'error': f'Prediction error: {e}'}), 500
    return jsonify({'prediction': prediction})

if __name__ == '__main__':
    # Use debug=False for production; True helps during development
    app.run(debug=False, host='0.0.0.0', port=5000)

