from flask import Flask, request, jsonify

app = Flask(__name__)

@app.route('/')
def home():
    return "✅ Flask app is running on localhost!"

@app.route('/webhook', methods=['POST'])
def webhook():
    req = request.get_json(force=True)

    # Debug print to see what Dialogflow sends
    print("👉 Received request from Dialogflow:")
    print(req)

    # Example: Extract the intent name
    intent_name = req.get('queryResult', {}).get('intent', {}).get('displayName', 'Unknown')

    # Send a simple text response back
    response = {
        'fulfillmentText': f"You triggered the intent: {intent_name}"
    }

    return jsonify(response)

if __name__ == '__main__':
    app.run(debug=True)
