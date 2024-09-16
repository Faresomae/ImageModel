from flask import Flask, request, jsonify
import tensorflow as tf
import numpy as np
from PIL import Image
import io
import h5py

app = Flask(__name__)

# Load the TFLite model (replace 'model_quantized.tflite' with the actual path to your model)
try:
    interpreter = tflite.Interpreter(model_path='model_quantized.tflite')
    interpreter.allocate_tensors()
except Exception as e:
    print(f"Failed to load TFLite model: {e}")
    raise

# Get input and output tensors
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Prediction function
def predict_from_model(img_array):
    try:
        # Set the input tensor
        interpreter.set_tensor(input_details[0]['index'], img_array)
        interpreter.invoke()

        # Get the output tensor
        predictions = interpreter.get_tensor(output_details[0]['index'])
        predicted_class = np.argmax(predictions, axis=1)
        return int(predicted_class[0])
    except Exception as e:
        print(f"Prediction error: {e}")
        raise

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

        # Adjust the shape of the input array if needed
        img_array = np.array(img_array, dtype=np.float32)

        prediction = predict_from_model(img_array)

        response = {
            "predicted_class": prediction,  # Adjust based on your model's output
        }

        return jsonify(response), 200
    except Exception as e:
        print(f"Error in prediction endpoint: {e}")
        return jsonify({"error": str(e)}), 500

# if __name__ == '__main__':
#     app.run(debug=True)
