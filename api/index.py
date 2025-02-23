from flask import Flask, request, jsonify
import requests

app = Flask(__name__)

@app.route('/generate', methods=['POST'])
def generate_response():
    try:
        # Get data from the request JSON
        data = request.json
        
        # Extract required parameters from the request; you can also add validations if needed
        hf_api_url = data.get('hf_api_url', '')
        hf_api_key = data.get('hf_api_key', '')
        model = data.get('model', '')
        user_input = data.get('user_input', '')
        
        if not all([hf_api_url, hf_api_key, model, user_input]):
            return jsonify({"error": "hf_api_url, hf_api_key, model, and user_input are required"}), 400

        # Define headers for the API call (including Connection: close if desired)
        headers = {
            "Authorization": f"Bearer {hf_api_key}",
            "Content-Type": "application/json",
            "Connection": "close"  # Optional: Force closing the connection after each request
        }

        # Construct the payload for the Hugging Face API call
        payload = {
            "model": model,
            "messages": [
                {
                    "role": "system",
                    "content": "You are a helpful and harmless assistant. You are Qwen developed by Alibaba. You should think step-by-step."
                },
                {
                    "role": "user",
                    "content": user_input
                }
            ],
            "temperature": 0.5,
            "max_tokens": 2048,
            "top_p": 0.7,
            "stream": False  # Disable streaming for easier handling
        }

        # Send the POST request to the provided Hugging Face API URL
        response = requests.post(hf_api_url, headers=headers, json=payload)

        # If the request was successful, return the API response as JSON
        if response.status_code == 200:
            return jsonify(response.json())
        else:
            # If there was an error, return the error message and status code
            return jsonify({"error": response.text}), response.status_code

    except Exception as e:
        # Catch any exceptions and return an error response
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
