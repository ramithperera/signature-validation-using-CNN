from flask import Flask, request, jsonify
from keras.models import load_model
import cv2
import numpy as np

app = Flask(__name__)

# Load the trained model
model = load_model('signature_validation_model.h5')


# Define the API endpoint
@app.route('/validate_signature', methods=['POST'])
def validate_signature():
    # Check if an image file was provided in the request
    if 'file' not in request.files:
        return jsonify({'error': 'No image file provided.'})

    file = request.files['file']

    # Read and preprocess the image
    image = cv2.imdecode(np.fromstring(file.read(), np.uint8), cv2.IMREAD_COLOR)
    image = cv2.resize(image, (128, 128))
    image = image.astype('float32') / 255.0
    image = np.expand_dims(image, axis=0)

    # Make predictions
    prediction = model.predict(image)[0][0]
    result = 'Forged' if prediction > 0.5 else 'Genuine'

    # Return the result to the user
    return jsonify({'result': result})


if __name__ == '__main__':
    # Change the port number as desired (default: 5000)
    port = 5000

    print(f'Server is running on port {port}')
    app.run(port=port)

