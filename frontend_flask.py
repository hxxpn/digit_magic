from flask import Flask, render_template, request, send_file
from PIL import Image
import io
import base64

app = Flask(__name__)


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/save', methods=['POST'])
def save():
    # Get the image data from the request
    image_data = request.json['imageData']

    # Remove the data URL prefix
    image_data = image_data.split(',')[1]

    # Decode the base64 image
    image = Image.open(io.BytesIO(base64.b64decode(image_data)))

    # Resize the image to 28x28
    image = image.resize((28, 28), Image.LANCZOS)

    # Convert to grayscale
    image = image.convert('L')

    # Save the image to a byte stream
    img_byte_arr = io.BytesIO()
    image.save(img_byte_arr, format='PNG')
    img_byte_arr.seek(0)

    # Send the file
    return send_file(img_byte_arr, mimetype='image/png', as_attachment=True, download_name='drawing.png')


if __name__ == '__main__':
    app.run(debug=True)
