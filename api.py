import os
import logging
from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename
from models.domain_classifier import ResearchDomainClassifier
from src.summary import ResearchPaperProcessor, main as process_paper

# Configure logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Create Flask app
app = Flask(__name__)

# Configuration
UPLOAD_FOLDER = 'D:\\Brieflet\\data\\pdf_files'
ALLOWED_EXTENSIONS = {'pdf'}

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Load pre-trained models
try:
    domain_classifier = ResearchDomainClassifier(domain_keywords_file='D:\\Brieflet\\models\\domain_keywords.json')
    logger.info("Domain classifier model loaded successfully")
except Exception as e:
    logger.error(f"Failed to load domain classifier: {e}")
    domain_classifier = None

try:
    paper_processor = ResearchPaperProcessor(pdf_path=None)
    logger.info("Summarizer model loaded successfully")
except Exception as e:
    logger.error(f"Failed to load summarizer: {e}")
    paper_processor = None

def allowed_file(filename):
    """Check if file has an allowed extension."""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/analyze', methods=['POST'])
def analyze_paper():
    """
    Endpoint to analyze a research paper:
    1. Upload PDF
    2. Classify domain
    3. Generate summary
    """
    # Check if file is present
    if 'file' not in request.files:
        return jsonify({
            'error': 'No file uploaded',
            'status': 'failure'
        }), 400

    file = request.files['file']

    # Check if filename is empty
    if file.filename == '':
        return jsonify({
            'error': 'No selected file',
            'status': 'failure'
        }), 400

    # Check file type
    if not allowed_file(file.filename):
        return jsonify({
            'error': 'Invalid file type. Only PDFs are allowed.',
            'status': 'failure'
        }), 400

    try:
        # Secure filename and save file
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        logger.info(f"File saved: {filepath}")

        # Process paper
        results = process_paper(filepath)

        # Classify domain (if model is loaded)
        domain = "Unknown"
        if domain_classifier:
            try:
                domain = domain_classifier.predict(results['sections'].get('abstract', ''))
                logger.info(f"Domain classified as: {domain}")
            except Exception as domain_error:
                logger.error(f"Domain classification failed: {domain_error}")

        # Generate summary (if model is loaded)
        if paper_processor:
            summary_result = paper_processor.generate_summary(results['sections'].get('abstract', ''))
            results['summary'] = summary_result
        else:
            results['summary'] = {
                'summary': 'Summary generation failed. Model not loaded.',
                'original_length': 0,
                'summary_length': 0
            }

        # Prepare response
        response = {
            'status': 'success',
            'filename': filename,
            'domain': domain,
            'sections': results['sections'],
            'summary': results['summary']
        }

        return jsonify(response), 200

    except Exception as e:
        logger.error(f"Processing error: {e}", exc_info=True)
        return jsonify({
            'error': str(e),
            'status': 'failure'
        }), 500

@app.route('/health', methods=['GET'])
def health_check():
    """Simple health check endpoint."""
    return jsonify({
        'status': 'healthy',
        'domain_classifier': domain_classifier is not None,
        'summarizer': paper_processor is not None
    }), 200

if __name__ == '__main__':
    app.run(debug=True, port=5000)
