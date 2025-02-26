from flask import Flask, request, jsonify, session
from hugchat import hugchat
from hugchat.login import Login
import secrets
import uuid

app = Flask(__name__)

# Generate a secure random secret key
app.secret_key = secrets.token_hex(16)

# Authenticate once and create a persistent ChatBot instance
email = 'becevofo@thetechnext.net'
password = 'Lumeth12#'
sign = Login(email, password)
cookies = sign.login()
chatbot = hugchat.ChatBot(cookies=cookies.get_dict())

# Dictionary to manage user-specific conversations
user_conversations = {}

@app.route('/generate', methods=['POST'])
def generate():
    data = request.json
    prompt = data.get('prompt')
    model_index = data.get('model_index', 0)
    system_prompt = data.get('system_prompt')

    if not prompt:
        return jsonify({'error': 'Prompt is required.'}), 400

    # Assign a unique user ID if not already present in the session
    if 'user_id' not in session:
        session['user_id'] = str(uuid.uuid4())

    user_id = session['user_id']

    # Retrieve or create a conversation ID for the user
    if user_id not in user_conversations:
        conversation_id = chatbot.new_conversation()
        user_conversations[user_id] = conversation_id
    else:
        conversation_id = user_conversations[user_id]

    try:
        # Switch to the desired model if necessary
        chatbot.change_model(model_index)
        # Set the system prompt if provided
        if system_prompt:
            chatbot.system_prompt = system_prompt
        # Generate response
        response = chatbot.chat(prompt, conversation_id=conversation_id)
        return jsonify({'response': response})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
