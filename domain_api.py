import os
import logging
from flask import Flask, request, jsonify
from models.domain_classifier import ResearchDomainClassifier

# Initialize the Flask app
app = Flask(__name__)

# Set up logging
logging.basicConfig(
    level=logging.DEBUG,  # Set the logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(),  # Output logs to the console
        logging.FileHandler("app.log")  # Output logs to a file
    ]
)

# Initialize and train the classifier dynamically
classifier = ResearchDomainClassifier()
pdf_directory = 'D:\\Brieflet\\data\\pdf_files'

# Train the model upon app startup
try:
    X_train, X_test, y_train, y_test = classifier.prepare_dataset(pdf_directory)
    classifier.train(X_train, y_train)
    logging.info("Model trained successfully.")
except Exception as e:
    logging.error(f"Error training the model: {e}")


@app.route("/")
def home():
    """
    Home route to check if the API is running.

    Returns:
        jsonify: A JSON response with a message confirming the API is running.
    """
    return jsonify({"message": "Research Domain Classifier API is running."})


@app.route("/classify", methods=["POST"])
def classify_pdf():
    """
    Classify the research domain of a PDF.

    This route accepts a PDF file uploaded via form-data, extracts the text, 
    and classifies it into a research domain.

    Returns:
        jsonify: A JSON response with the predicted domain or error message.
    """
    if 'file' not in request.files:
        logging.warning("No file provided.")
        return jsonify({"error": "No file provided."}), 400

    pdf_file = request.files['file']
    if pdf_file.filename == '':
        logging.warning("No selected file.")
        return jsonify({"error": "No selected file."}), 400

    if not pdf_file.filename.endswith('.pdf'):
        logging.warning(f"Invalid file format: {pdf_file.filename}. Only PDF files are supported.")
        return jsonify({"error": "Invalid file format. Only PDF files are supported."}), 400

    try:
        # Save the uploaded PDF temporarily
        temp_path = os.path.join("temp", pdf_file.filename)
        os.makedirs("temp", exist_ok=True)
        pdf_file.save(temp_path)
        logging.info(f"PDF file {pdf_file.filename} saved temporarily.")

        # Extract and classify the PDF text
        pdf_text = ResearchDomainClassifier.pdf_to_text(temp_path)
        domain = classifier.predict(pdf_text)
        logging.info(f"Predicted domain for {pdf_file.filename}: {domain}")

        # Cleanup temp file
        os.remove(temp_path)
        logging.info(f"Temporary file {pdf_file.filename} removed.")

        return jsonify({"domain": domain})
    except Exception as e:
        logging.error(f"Error processing file {pdf_file.filename}: {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/predict", methods=["POST"])
def predict_text():
    """
    Predict the research domain from text input.

    This route accepts a JSON request containing text and predicts its
    research domain.

    Returns:
        jsonify: A JSON response with the predicted domain or error message.
    """
    data = request.get_json()
    if not data or 'text' not in data:
        logging.warning("No text provided in the request.")
        return jsonify({"error": "No text provided."}), 400

    text = data['text']
    try:
        domain = classifier.predict(text)
        logging.info(f"Predicted domain for text: {domain}")
        return jsonify({"domain": domain})
    except Exception as e:
        logging.error(f"Error predicting domain for the provided text: {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/health", methods=["GET"])
def health():
    """
    Health check route to ensure the API is operational.

    Returns:
        jsonify: A JSON response indicating the API health status.
    """
    try:
        # If the classifier is trained, consider the API healthy
        classifier_health = "Healthy" if classifier else "Unhealthy"
        return jsonify({"status": "API is healthy", "classifier_status": classifier_health}), 200
    except Exception as e:
        logging.error(f"Error in health check: {e}")
        return jsonify({"status": "API is unhealthy", "error": str(e)}), 500


if __name__ == "__main__":
    app.run(debug=True)
