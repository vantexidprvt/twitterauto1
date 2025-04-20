from flask import Flask, request, jsonify
from gradio_client import Client
import requests
import os

app = Flask(__name__)
client = Client("black-forest-labs/FLUX.1-schnell")

@app.route('/generate', methods=['POST'])
def generate_image():
    try:
        data = request.json

        prompt = data.get("prompt", "")
        seed = data.get("seed", 42)
        randomize_seed = data.get("randomize_seed", True)
        width = data.get("width", 1024)
        height = data.get("height", 1024)
        num_inference_steps = data.get("num_inference_steps", 4)

        # Step 1: Generate the image (returns a tuple, extract only file path)
        result_tuple = client.predict(
            prompt=prompt,
            seed=seed,
            randomize_seed=randomize_seed,
            width=width,
            height=height,
            num_inference_steps=num_inference_steps,
            api_name="/infer"
        )
        result = result_tuple[0]  # Get only the file path

        # Step 2: Upload to tmpfiles.org
        if os.path.isfile(result):
            with open(result, 'rb') as f:
                upload_response = requests.post(
                    'https://tmpfiles.org/api/v1/upload',
                    files={'file': f}
                )

            # Step 3: Parse the response
            upload_json = upload_response.json()
            tmp_url = upload_json.get("data", {}).get("url")

            # Step 4: Delete the local file
            os.remove(result)

            if tmp_url:
                return jsonify({"image_url": tmp_url})
            else:
                return jsonify({"error": "Upload failed"}), 500

        else:
            return jsonify({"error": "Generated file not found"}), 500

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == '__main__':
    app.run(debug=True)
