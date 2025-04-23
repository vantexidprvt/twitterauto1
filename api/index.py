from flask import Flask, request, jsonify
from gradio_client import Client, file
import requests
import os

app = Flask(__name__)
client = Client("AIRI-Institute/HairFastGAN")

TMPFILES_UPLOAD_URL = "https://tmpfiles.org/api/v1/upload"

@app.route('/process-image', methods=['POST'])
def process_image():
    data = request.get_json()
    image_url = data.get('image_url')
    if not image_url:
        return jsonify({"error": "No image URL provided"}), 400

    local_path = None
    try:
        # 1) Run your Gradio model
        local_path = client.predict(
            img=file(image_url),
            align=["Face", "Shape", "Color"],
            api_name="/resize_inner"
        )
        # local_path is something like "/tmp/tmp-1234abcd.png"

        # 2) Upload that file to tmpfiles.org
        with open(local_path, 'rb') as f:
            files = {'file': (os.path.basename(local_path), f)}
            resp = requests.post(TMPFILES_UPLOAD_URL, files=files)

        if resp.status_code != 200:
            return jsonify({
                "error": "Failed to upload to tmpfiles.org",
                "details": resp.text
            }), 502

        # 3) Parse tmpfiles.org response
        try:
            result = resp.json()
            tmp_link = result.get('url') or result.get('link') or result
        except ValueError:
            tmp_link = resp.text.strip()

        # 4) Return the link
        return jsonify({
            "tmpfiles_link": tmp_link
        }), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500

    finally:
        # 5) Clean up the local file if it exists
        if local_path and os.path.isfile(local_path):
            try:
                os.remove(local_path)
            except OSError:
                pass  # optionally log this
