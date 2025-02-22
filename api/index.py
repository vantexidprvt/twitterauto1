from flask import Flask, request, jsonify
import requests

app = Flask(__name__)

# Hugging Face API URL and API key
HF_API_URL = "https://api-inference.huggingface.co/models/Qwen/QwQ-32B-Preview/v1/chat/completions"
HF_API_KEY = "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"

# Define headers for the API call (including Connection: close if desired)
headers = {
    "Authorization": f"Bearer {HF_API_KEY}",
    "Content-Type": "application/json",
    "Connection": "close"  # Optional: Force closing the connection after each request
}

@app.route('/generate', methods=['POST'])
def generate_response():
    try:
        # Extract the user's input from the JSON body of the request
        user_input = request.json.get('user_input', '')

        # Construct the payload for the Hugging Face API call
        payload = {
            "model": "Qwen/QwQ-32B-Preview",
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

        # Send the POST request to the Hugging Face API
        response = requests.post(HF_API_URL, headers=headers, json=payload)

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
