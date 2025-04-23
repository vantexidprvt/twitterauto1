import requests
from flask import Flask, request, jsonify
from gradio_client import Client, file

# Initialize Flask app
app = Flask(__name__)

# Initialize the Gradio client
client = Client("AIRI-Institute/HairFastGAN")

# Function to upload the file to tmpfiles.org
def upload_to_tmpfiles(file_content):
    url = "https://tmpfiles.org/api/v1/upload"
    files = {
        'file': ('image.webp', file_content, 'image/webp')  # Set MIME type appropriately
    }
    response = requests.post(url, files=files)
    if response.status_code == 200:
        return response.json().get('url')
    else:
        raise Exception(f"File upload failed: {response.status_code} - {response.text}")

@app.route('/process-image', methods=['POST'])
def process_image():
    # Get image URL from the request
    data = request.get_json()
    image_url = data.get('image_url')

    if not image_url:
        return jsonify({"error": "No image URL provided"}), 400

    try:
        # Fetch the image using Gradio client (without any resizing)
        image_data = client.predict(
            img=file(image_url),
            api_name="/predict"
        )

        # Assuming the result is returned as binary content
        image_content = image_data[0]  # Adjust this according to the actual response structure

        # Upload the image to tmpfiles.org
        file_link = upload_to_tmpfiles(image_content)

        # Return the file link
        return jsonify({"file_link": file_link}), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
