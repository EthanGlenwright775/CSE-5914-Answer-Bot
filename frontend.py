from flask import Flask, render_template, request

app = Flask('frontend.py')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/send_message', methods=['POST'])
def send_message():
    message = request.form['message']
    # Process the message (e.g., send to chatbot model)
    # For demonstration, let's echo the message back
    return {'message': "thanks for saying that!"}

if __name__ == '__main__':
    app.run(debug=True)