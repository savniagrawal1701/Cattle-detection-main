import os
import io
import torch
import cv2
import numpy as np
from PIL import Image
from flask import Flask, request, render_template, send_file

app = Flask(__name__)

# Load Model (Optimized for Render's low RAM)
model = torch.hub.load("ultralytics/yolov5", "custom", path="weights/yolov5s.pt", force_reload=False)
model.conf = 0.4

@app.route('/')
def index():
    return '''
    <h1>Cattle Detection System</h1>
    <form action="/predict" method="post" enctype="multipart/form-data">
      <input type="file" name="file">
      <input type="submit" value="Upload and Detect">
    </form>
    '''

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return "No file uploaded", 400
    
    file = request.files['file']
    img_bytes = file.read()
    img = Image.open(io.BytesIO(img_bytes))
    
    # Perform Detection
    results = model(img)
    
    # Render results on the image
    results.render()  # This updates results.ims with boxes
    
    # Convert back to image to send to browser
    res_img = Image.fromarray(results.ims[0])
    byte_io = io.BytesIO()
    res_img.save(byte_io, 'JPEG')
    byte_io.seek(0)
    
    return send_file(byte_io, mimetype='image/jpeg')

if __name__ == "__main__":
    # Render provides a PORT environment variable
    port = int(os.environ.get("PORT", 10000))
    app.run(host='0.0.0.0', port=port)
