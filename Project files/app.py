from flask import Flask, render_template, request
import os
from werkzeug.utils import secure_filename
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model  # ✅ Use this for .h5 model

app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# ✅ Load the Keras model properly
model = load_model('model/pollen_classification_model (2).h5')

# ✅ Class labels
label_map = {
    0: "arecaceae", 1: "arrabidaea", 2: "cecropia", 3: "chromolaena",
    4: "combretum", 5: "croton", 6: "dipteryx", 7: "faramea",
    8: "hyptis", 9: "matayba", 10: "mimosa", 11: "protium",
    12: "senegalia", 13: "serjania", 14: "tridax"
}

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return 'No file part'

    file = request.files['file']
    if file.filename == '':
        return 'No selected file'

    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)

    # ✅ Preprocess the image
    img = Image.open(filepath).resize((224, 224))  # Adjust size to model input
    img = img.convert('RGB')  # Ensure it's 3-channel
    img = np.array(img) / 255.0
    img = np.expand_dims(img, axis=0)  # Add batch dimension

    # ✅ Make prediction
    prediction = model.predict(img)
    predicted_class = label_map[np.argmax(prediction)]

    return render_template('prediction.html', prediction=predicted_class)

@app.route('/logout')
def logout():
    return render_template('logout.html')

if __name__ == '__main__':
    if not os.path.exists(UPLOAD_FOLDER):
        os.makedirs(UPLOAD_FOLDER)
    app.run(debug=True)
