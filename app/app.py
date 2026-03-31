import os
import sys

# Ensure src is in python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from app.region_detector import detect_fire_region
from flask import Flask, render_template, request, jsonify, url_for
from werkzeug.utils import secure_filename
from src.detection.image_detection import ImageDetector
from src.utils.helpers import ensure_dir

app = Flask(__name__)

# Configurations
app.config['UPLOAD_FOLDER'] = 'app/static/uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB

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
        return jsonify({'error': 'No file uploaded'}), 400

    file = request.files['file']

    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    if not allowed_file(file.filename):
        return jsonify({'error': 'Invalid file type'}), 400

    filename = secure_filename(file.filename)

    input_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(input_path)

    output_filename = "pred_" + filename
    output_path = os.path.join(app.config['UPLOAD_FOLDER'], output_filename)

    try:
        prediction, has_fire, detections = detector.detect(
            input_path,
            output_path=output_path
        )
        if has_fire:
            detect_fire_region(input_path)

    except Exception as e:
        return jsonify({'error': str(e)}), 500

    result_url = url_for(
        'static',
        filename='uploads/' + output_filename
    )

    return jsonify({
        'success': True,
        'prediction': prediction,
        'has_fire': bool(has_fire),
        'detections_count': len(detections),
        'result_image_url': result_url
    })


if __name__ == '__main__':
    app.run(debug=True, port=5000)