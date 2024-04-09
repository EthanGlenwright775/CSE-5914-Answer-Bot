from flask import Flask, render_template, request
from rephraser import rephrase
import nltk
import os,tempfile

# Create the pipe
tmpdir = tempfile.mkdtemp()
fifo_path = os.path.join(tmpdir, 'frontToModel')
os.mkfifo(fifo_path)

nltk.download('punkt', quiet=True)
from nltk.tokenize import sent_tokenize

app = Flask('frontend.py')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/send_message', methods=['POST'])
def send_message():
    with open(fifo_path, 'w') as fifo:
        message = message = request.form['message']
        fifo.write(message)

    with open(fifo_path, 'r') as fifo:
        message = fifo.read()
        return {'message': message}
    # Process the message (e.g., send to chatbot model)
    # phrases = sent_tokenize(rephrase(message))
    # For demonstration, let's echo the message back

if __name__ == '__main__':
    app.run(debug=True)

# Send context for ./frontToModel pipe
# Send question for ./frontToModel pipe
