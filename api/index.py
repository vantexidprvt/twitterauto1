from flask import Flask, request, jsonify, session
from hugchat import hugchat
from hugchat.login import Login
import secrets
import uuid

app = Flask(__name__)

# Secure session key
app.secret_key = secrets.token_hex(16)

# Dictionary to store chatbot instances per user session
user_sessions = {}

def get_chatbot_instance(email, password, session_id):
    """Retrieve or create a chatbot instance for the user session."""
    if session_id in user_sessions:
        return user_sessions[session_id]  # Return existing chatbot instance

    # Authenticate and create chatbot
    sign = Login(email, password)
    cookies = sign.login()
    chatbot = hugchat.ChatBot(cookies=cookies.get_dict())

    # Store chatbot for this session
    user_sessions[session_id] = chatbot
    return chatbot

@app.route('/generate', methods=['POST'])
def generate():
    """Generate AI response for a given user session."""
    data = request.json
    prompt = data.get('prompt')
    email = data.get('email')
    password = data.get('password')
    system_prompt = data.get('system_prompt')

    if not all([prompt, email, password]):
        return jsonify({'error': 'Prompt, email, and password are required.'}), 400

    # Assign a unique session ID for each user
    if 'session_id' not in session:
        session['session_id'] = str(uuid.uuid4())  # Generate a unique ID

    session_id = session['session_id']

    # Retrieve or create a chatbot instance for this session
    chatbot = get_chatbot_instance(email, password, session_id)

    try:
        # Start a new conversation if not already started
        conversation_id = chatbot.new_conversation()

        # Set system prompt if provided
        if system_prompt:
            chatbot.system_prompt = system_prompt

        # Generate response
        response = chatbot.chat(prompt, conversation_id=conversation_id)

        return jsonify({'response': response})

    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/reset_chat', methods=['POST'])
def reset_chat():
    """Reset the chat history for the current user session."""
    session_id = session.get('session_id')

    if session_id and session_id in user_sessions:
        chatbot = user_sessions[session_id]
        chatbot.delete_conversation()  # Clear chat history
        del user_sessions[session_id]  # Remove session

        return jsonify({'message': 'Chat history cleared.'})

    return jsonify({'error': 'No active chat found.'}), 400

if __name__ == '__main__':
    app.run(debug=True)
