import os
import pandas as pd
import PyPDF2
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report


class ResearchDomainClassifier:
    def __init__(self):
        self.vectorizer = TfidfVectorizer(stop_words='english')
        self.classifier = MultinomialNB()

    @staticmethod
    def pdf_to_text(pdf_path):
        """Extract text from a PDF file."""
        text = ""
        with open(pdf_path, 'rb') as file:
            reader = PyPDF2.PdfReader(file)
            for page_num in range(len(reader.pages)):
                page = reader.pages[page_num]
                text += page.extract_text()
        return text

    @staticmethod
    def assign_domain_label(pdf_file_name, pdf_text):
        """
        Assign domain label to a given file based on file name and content.
        """
        domain_keywords = {
            # Domain Keywords
            "Artificial Intelligence": ["AI", "machine learning", "deep learning", "neural network", "natural language processing"],
            "Biology": ["DNA", "genome", "biochemistry", "microbiology", "protein"],
            "Physics": ["quantum mechanics", "thermodynamics", "particle physics", "optics"],
            "Chemistry": ["chemical reaction", "organic chemistry", "biochemistry"],
            "Mathematics": ["calculus", "algebra", "geometry", "statistics"],
            "Computer Science": ["programming", "algorithm", "data structure", "cybersecurity"],
            # Add other domains...
        }

        # Check file name for domain-specific keywords
        for domain, keywords in domain_keywords.items():
            if any(keyword.lower() in pdf_file_name.lower() for keyword in keywords):
                return domain

        # Check text content for domain-specific keywords
        for domain, keywords in domain_keywords.items():
            if any(keyword.lower() in pdf_text.lower() for keyword in keywords):
                return domain

        return "Unknown"

    def prepare_dataset(self, pdf_dir):
        """
        Prepare dataset from PDF files in a directory.
        """
        data = []

        # Process all PDFs in the directory
        for pdf_file in os.listdir(pdf_dir):
            if pdf_file.endswith('.pdf'):
                pdf_path = os.path.join(pdf_dir, pdf_file)
                pdf_text = self.pdf_to_text(pdf_path)
                domain = self.assign_domain_label(pdf_file, pdf_text)
                data.append({'text': pdf_text, 'domain': domain})

        # Create DataFrame
        df = pd.DataFrame(data)

        # Vectorize text
        X = self.vectorizer.fit_transform(df['text'])
        y = df['domain']

        # Split dataset
        return train_test_split(X, y, test_size=0.2, random_state=42)

    def train(self, X_train, y_train):
        """Train the classifier with the given dataset."""
        self.classifier.fit(X_train, y_train)

    def evaluate(self, X_test, y_test):
        """Evaluate the classifier's performance."""
        y_pred = self.classifier.predict(X_test)
        return classification_report(y_test, y_pred)

    def predict(self, text):
        """Predict the domain of a given text."""
        vectorized_text = self.vectorizer.transform([text])
        return self.classifier.predict(vectorized_text)[0]

    @classmethod
    def load_model(cls, model_path):
        # For future use if needed for loading a pre-trained model
        pass


# Flask API Implementation
from flask import Flask, request, jsonify

app = Flask(__name__)

# Initialize and train the classifier dynamically
classifier = ResearchDomainClassifier()
pdf_directory = 'D:\\research_summarizer\\data\\pdf_files'

# Train the model upon app startup
try:
    X_train, X_test, y_train, y_test = classifier.prepare_dataset(pdf_directory)
    classifier.train(X_train, y_train)
    print("Model trained successfully.")
except Exception as e:
    print(f"Error training the model: {e}")


@app.route("/")
def home():
    return jsonify({"message": "Research Domain Classifier API is running."})


@app.route("/classify", methods=["POST"])
def classify_pdf():
    """
    Classify the research domain of a PDF.
    Expects a PDF file to be uploaded via form-data.
    """
    if 'file' not in request.files:
        return jsonify({"error": "No file provided."}), 400

    pdf_file = request.files['file']
    if pdf_file.filename == '':
        return jsonify({"error": "No selected file."}), 400

    if not pdf_file.filename.endswith('.pdf'):
        return jsonify({"error": "Invalid file format. Only PDF files are supported."}), 400

    try:
        # Save the uploaded PDF temporarily
        temp_path = os.path.join("temp", pdf_file.filename)
        os.makedirs("temp", exist_ok=True)
        pdf_file.save(temp_path)

        # Extract and classify the PDF text
        pdf_text = ResearchDomainClassifier.pdf_to_text(temp_path)
        domain = classifier.predict(pdf_text)

        # Cleanup temp file
        os.remove(temp_path)

        return jsonify({"domain": domain})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/predict", methods=["POST"])
def predict_text():
    """
    Predict the research domain from text input.
    Expects JSON with a "text" field.
    """
    data = request.get_json()
    if not data or 'text' not in data:
        return jsonify({"error": "No text provided."}), 400

    text = data['text']
    try:
        domain = classifier.predict(text)
        return jsonify({"domain": domain})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(debug=True)
