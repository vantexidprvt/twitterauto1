from flask import Flask, request, jsonify
from gradio_client import Client, file

# Initialize Flask app
app = Flask(__name__)

# Initialize the Gradio client
client = Client("AIRI-Institute/HairFastGAN")

@app.route('/process-image', methods=['POST'])
def process_image():
    # Get image URL from the request
    data = request.get_json()
    image_url = data.get('image_url')

    if not image_url:
        return jsonify({"error": "No image URL provided"}), 400

    try:
        # Use Gradio client to predict
        response = client.predict(
            img=file(image_url),
            align=["Face", "Shape", "Color"],
            api_name="/resize_inner"
        )

        # Return the response from Gradio
        return jsonify({"result": response}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
