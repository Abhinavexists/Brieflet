import os
import logging
from flask import Flask, request, jsonify
from models.domain_classifier import ResearchDomainClassifier
from werkzeug.utils import secure_filename

# Configure logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Create Flask app
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'D:\\Brieflet\\data\\pdf_files'

# Load pre-trained domain classifier model
try:
    domain_classifier = ResearchDomainClassifier(domain_keywords_file='D:\\Brieflet\\models\\domain_keywords.json')
    logger.info("Domain classifier model loaded successfully")
except Exception as e:
    logger.error(f"Failed to load domain classifier: {e}")
    domain_classifier = None

@app.route('/classify', methods=['POST'])
def classify():
    if 'file' not in request.files:
        logger.error("No file part in the request")
        return jsonify({
            'error': 'No file part in the request',
            'status': 'failure'
        }), 400

    file = request.files['file']
    if file.filename == '':
        logger.error("No file selected for uploading")
        return jsonify({
            'error': 'No file selected for uploading',
            'status': 'failure'
        }), 400

    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)

        try:
            with open(file_path, 'r') as f:
                text = f.read()

            if not text:
                logger.error("File is empty")
                return jsonify({
                    'error': 'File is empty',
                    'status': 'failure'
                }), 400

            # Assuming domain_classifier is already loaded and fitted
            domain = domain_classifier.predict(text)
            return jsonify({
                'status': 'success',
                'domain': domain
            }), 200
        except Exception as e:
            logger.error(f"Error during classification: {e}")
            return jsonify({
                'error': 'Internal server error',
                'status': 'failure'
            }), 500
    else:
        logger.error("Invalid file type")
        return jsonify({
            'error': 'Invalid file type',
            'status': 'failure'
        }), 400

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in {'txt', 'pdf', 'docx'}

@app.route('/health', methods=['GET'])
def health_check():
    """Simple health check endpoint."""
    return jsonify({
        'status': 'healthy',
        'domain_classifier': domain_classifier is not None
    }), 200

if __name__ == '__main__':
    app.run(debug=True, port=5002)