import cv2
import numpy as np
from flask import Flask, request, jsonify
import requests
import io
import uuid  # Import the uuid module for generating a random filename

app = Flask(__name__)

def process_images(a1_url, a2_url, a3_url):
    try:
        # Load images from URLs
        a1_response = requests.get(a1_url)
        a1_content = io.BytesIO(a1_response.content)
        a1_image = cv2.imdecode(np.frombuffer(a1_content.read(), np.uint8), cv2.IMREAD_GRAYSCALE)

        a2_response = requests.get(a2_url)
        a2_content = io.BytesIO(a2_response.content)
        a2_image = cv2.imdecode(np.frombuffer(a2_content.read(), np.uint8), cv2.IMREAD_UNCHANGED)

        a3_response = requests.get(a3_url)
        a3_content = io.BytesIO(a3_response.content)
        a3_image = cv2.imdecode(np.frombuffer(a3_content.read(), np.uint8), cv2.IMREAD_UNCHANGED)

        # Ensure a1_image and a2_image have the same dimensions
        height, width = a1_image.shape
        a2_image = cv2.resize(a2_image, (width, height))

        # Create a mask for a2_image based on white areas in a1_image
        mask = (a1_image == 255).astype(np.uint8)

        # Create an inverse mask
        mask_inv = 1 - mask

        # Copy a2_image to new_image using the mask
        new_image = a2_image.copy()
        new_image[mask_inv == 1] = 0  # Set non-white areas to 0 (black)

        # Combine new_image and a3_image using bitwise AND
        result_image = cv2.bitwise_and(a3_image, a3_image, mask=mask_inv)
        result_image = cv2.bitwise_or(result_image, new_image)

        # Generate a random filename for the result image
        result_filename = str(uuid.uuid4()) + '.png'

        # Save the result image with the random filename
        cv2.imwrite(result_filename, result_image)

        return result_filename  # Return the random filename

    except Exception as e:
        return str(e)

@app.route('/process_images', methods=['GET'])
def api_process_images():
    a1_url = request.args.get('a1')
    a2_url = request.args.get('a2')
    a3_url = request.args.get('a3')

    result_filename = process_images(a1_url, a2_url, a3_url)

    return jsonify({"result_filename": result_filename})  # Return the random filename

if __name__ == '__main__':
    app.run(host='0.0.0.0')
