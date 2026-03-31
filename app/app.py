import os
import sys
# Ensure src is in python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from flask import Flask, render_template, request, jsonify, url_for
from werkzeug.utils import secure_filename
from src.detection.image_detection import ImageDetector
from src.utils.config import CONFIG
from src.utils.helpers import ensure_dir

app = Flask(__name__)
# Configurations
app.config['UPLOAD_FOLDER'] = 'app/static/uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024 # 16MB

# Ensure upload directory exists
ensure_dir(app.config['UPLOAD_FOLDER'])

# Initialize detector globally
detector = ImageDetector()

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/detect', methods=['POST'])
def detect_fire():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        input_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(input_path)
        
        # Run Detection
        output_filename = "pred_" + filename
        output_path = os.path.join(app.config['UPLOAD_FOLDER'], output_filename)
        
        # Ensure model is ready (safe check)
        try:
             _, has_fire, detections = detector.detect(input_path, output_path=output_path)
        except Exception as e:
             return jsonify({'error': str(e)}), 500
             
        # Build Response
        result_url = url_for('static', filename='uploads/' + output_filename)
        
        return jsonify({
            'success': True,
            'has_fire': bool(has_fire),
            'detections': len(detections),
            'result_image_url': result_url
        })
        
    return jsonify({'error': 'Invalid file type'}), 400

if __name__ == '__main__':
    app.run(debug=True, port=5000)
