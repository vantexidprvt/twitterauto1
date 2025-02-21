from flask import Flask, request, jsonify
from huggingface_hub import InferenceClient

app = Flask(__name__)

# Initialize the InferenceClient with your Hugging Face API key
client = InferenceClient(api_key="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")

# Define the home route
@app.route('/')
def home():
    return 'Hello, World!'

# Define the about route
@app.route('/about')
def about():
    return 'About'

# Define the API endpoint for generating responses
@app.route('/generate', methods=['POST'])
def generate():
    # Extract user input from the request
    user_input = request.json.get('user_input', '')
    
    # Define the conversation context
    messages = [
        {"role": "system", "content": "You are a helpful and harmless assistant. You are Qwen developed by Alibaba. You should think step-by-step."},
        {"role": "user", "content": user_input}
    ]
    
    # Generate a response using the Hugging Face model
    response = client.chat_completions(
        model="Qwen/QwQ-32B-Preview",
        messages=messages,
        temperature=0.5,
        max_tokens=2048,
        top_p=0.7
    )
    
    # Extract the generated message from the response
    generated_message = response['choices'][0]['message']['content']
    
    # Return the generated message as a JSON response
    return jsonify({'response': generated_message})

if __name__ == '__main__':
    app.run(debug=True)
