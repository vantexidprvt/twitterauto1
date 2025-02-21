from flask import Flask, request, jsonify
import requests

app = Flask(__name__)

# Hugging Face API URL and Headers
HF_API_URL = "https://api-inference.huggingface.co/models/Qwen/QwQ-32B-Preview/v1/chat/completions"
HF_API_KEY = "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"

headers = {
    "Authorization": f"Bearer {HF_API_KEY}",
    "Content-Type": "application/json"
}

@app.route('/generate', methods=['POST'])
def generate_response():
    try:
        # Get user input from JSON request
        user_input = request.json.get('user_input', '')

        # Construct the payload
        payload = {
            "model": "Qwen/QwQ-32B-Preview",
            "messages": [
                { "role": "system", "content": "You are a helpful and harmless assistant. You are Qwen developed by Alibaba. You should think step-by-step." },
                { "role": "user", "content": user_input }
            ],
            "temperature": 0.5,
            "max_tokens": 2048,
            "top_p": 0.7,
            "stream": False  # Disable streaming for easier handling
        }

        # Make the request to Hugging Face API
        response = requests.post(HF_API_URL, headers=headers, json=payload)

        # Check for successful response
        if response.status_code == 200:
            return jsonify(response.json())  # Return API response
        else:
            return jsonify({"error": response.text}), response.status_code

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == '__main__':
    app.run(debug=True)
