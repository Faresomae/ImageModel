from flask import Flask, request, jsonify
# import tensorflow as tf
# import numpy as np
from PIL import Image
import io
import pickle
import numpy as np

app = Flask(__name__)

# Load the model (replace 'path_to_your_model.h5' with the actual path to your model)
# model = tf.keras.models.load_model('imagemodel.h5')
with open('my_model.h5', 'rb') as file:
    model = pickle.load(file)
# Prediction function
def predict_from_model(model, img_array):
    predictions = model.predict(img_array)
    predicted_class = np.argmax(predictions, axis=1)
    return int(predicted_class[0])

# Prediction endpoint
@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({"message": "No file part in the request"}), 400

    file = request.files['file']
    
    if file.filename == '':
        return jsonify({"message": "No selected file"}), 400

    try:
        img = Image.open(io.BytesIO(file.read()))
        img = img.resize((128, 128))  # Resize image as required by your model
        img_array = np.array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = img_array / 255.0  # Normalize image array

        prediction = predict_from_model(model, img_array)
        
        response = {
            "predicted_class": prediction,  # Adjust based on your model's output
        }
        
        return jsonify(response), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == 'main':
    app.run(debug=True)
