from flask import Flask, request, jsonify
import requests
import os
from gradio_client import Client, file

app = Flask(__name__)

# Step 1: Upload file to tmpfiles.org
def upload_file_to_tmpfiles(file_path):
    upload_url = "https://tmpfiles.org/api/v1/upload"
    with open(file_path, 'rb') as f:
        response = requests.post(upload_url, files={'file': f})
    if response.status_code == 200:
        return response.json().get('url')
    else:
        return None

# Step 2: Process the uploaded file with HairFastGAN and get resized image
def process_file_with_gradio(image_url):
    client = Client("AIRI-Institute/HairFastGAN")
    try:
        result = client.predict(
            img=file(image_url),
            align=["Face", "Shape", "Color"],
            api_name="/resize_inner"
        )
        return result
    except Exception as e:
        return {"error": str(e)}

# Step 3: Delete temporary file from the server
def delete_temp_file(temp_file_path):
    try:
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)
            print(f"Temporary file {temp_file_path} deleted successfully.")
        else:
            print(f"Temporary file {temp_file_path} does not exist.")
    except Exception as e:
        print(f"Error deleting file: {str(e)}")

# Flask API Route
@app.route('/process-image', methods=['POST'])
def process_image():
    # Step 1: Get image URL from the request
    data = request.get_json()
    image_url = data.get('image_url')

    if not image_url:
        return jsonify({"error": "No image URL provided"}), 400

    # Step 2: Process the image using HairFastGAN model
    try:
        processing_result = process_file_with_gradio(image_url)

        if 'error' in processing_result:
            return jsonify({"error": processing_result['error']}), 500

        resized_image_path = processing_result.get('result')

        if resized_image_path:
            # Step 3: Upload the resized image to tmpfiles.org
            resized_image_url = upload_file_to_tmpfiles(resized_image_path)
            if resized_image_url:
                # Step 4: Delete the temporary file
                delete_temp_file(resized_image_path)
                return jsonify({"resized_image_url": resized_image_url}), 200
            else:
                return jsonify({"error": "Failed to upload resized image."}), 500
        else:
            return jsonify({"error": "Resized image path not found."}), 500

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == '__main__':
    app.run(debug=True)
