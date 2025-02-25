from flask import Flask, request, jsonify
from hugchat import hugchat
from hugchat.login import Login

app = Flask(__name__)

def generate_response(prompt_input, email, passwd, model_index=0, system_prompt=None):
    # Hugging Face Login
    sign = Login(email, passwd)
    cookies = sign.login()
    # Create ChatBot
    chatbot = hugchat.ChatBot(cookies=cookies.get_dict())
    # Retrieve available models
    models = chatbot.get_available_llm_models()
    # Check if the provided model_index is within the range of available models
    if 0 <= model_index < len(models):
        # Create a new conversation and switch to the desired model
        chatbot.new_conversation(switch_to=True, modelIndex=model_index)
    else:
        print("Invalid model index. Using the default model.")
    # Set the system prompt if provided
    if system_prompt:
        chatbot.system_prompt = system_prompt
    # Generate response
    response = chatbot.chat(prompt_input)
    # Ensure the response is fully processed before deleting the conversation
    response_text = str(response)  # Convert response to string to process it
    # Delete the current conversation to clear chat history
    chatbot.delete_conversation()
    return response_text

@app.route('/generate', methods=['POST'])
def generate():
    data = request.json
    prompt = data.get('prompt')
    email = data.get('email')
    password = data.get('password')
    model_index = data.get('model_index', 0)
    system_prompt = data.get('system_prompt', None)

    if not all([prompt, email, password]):
        return jsonify({'error': 'Prompt, email, and password are required.'}), 400

    try:
        response = generate_response(prompt, email, password, model_index, system_prompt)
        return jsonify({'response': response})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
