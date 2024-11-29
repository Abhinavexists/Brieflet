import os
import json
import pickle
import PyPDF2
import pandas as pd
import logging
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report

# Check if logger exists, if not create one
logger = logging.getLogger('ResearchDomainClassifier')
if not logger.hasHandlers():
    # Create file handler for logging to domain.log
    file_handler = logging.FileHandler('domain.log')
    file_handler.setLevel(logging.INFO)
    
    # Create console handler for logging to console
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    
    # Create formatter and add it to handlers
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    
    # Add handlers to the logger
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    # Set the logging level
    logger.setLevel(logging.INFO)

class ResearchDomainClassifier:
    def __init__(self, domain_keywords_file='domain_keywords.json'):
        """
        Initialize classifier with domain keywords and models.
        """
        logger.info("Initializing ResearchDomainClassifier")
        # Load domain keywords from JSON file
        self.domain_keywords = self.load_domain_keywords(domain_keywords_file)
        self.vectorizer = TfidfVectorizer(stop_words='english')
        self.classifier = MultinomialNB()

    @staticmethod
    def pdf_to_text(pdf_path):
        """
        Extract text from a PDF file.
        """
        logger.info(f"Extracting text from PDF: {pdf_path}")
        text = ""
        with open(pdf_path, 'rb') as file:
            reader = PyPDF2.PdfReader(file)
            for page_num in range(len(reader.pages)):
                page = reader.pages[page_num]
                text += page.extract_text()
        return text

    def assign_domain_label(self, pdf_file_name, pdf_text):
        """
        Assign domain label to a given file based on file name and content.
        """
        logger.info(f"Assigning domain label for file: {pdf_file_name}")
        # Check file name for domain-specific keywords
        for domain, keywords in self.domain_keywords.items():
            if any(keyword.lower() in pdf_file_name.lower() for keyword in keywords):
                logger.info(f"Domain '{domain}' assigned based on file name.")
                return domain

        # Check text content for domain-specific keywords
        for domain, keywords in self.domain_keywords.items():
            if any(keyword.lower() in pdf_text.lower() for keyword in keywords):
                logger.info(f"Domain '{domain}' assigned based on file content.")
                return domain

        logger.warning(f"Domain not found for file: {pdf_file_name}. Assigned 'Unknown'.")
        return "Unknown"

    @staticmethod
    def load_domain_keywords(json_file):
        """
        Load domain keywords from a JSON file.
        """
        logger.info(f"Loading domain keywords from: {json_file}")
        with open(json_file, 'r') as file:
            return json.load(file)

    def prepare_dataset(self, pdf_dir):
        """
        Prepare dataset from PDF files in a directory.
        """
        logger.info(f"Preparing dataset from PDF files in directory: {pdf_dir}")
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
        """
        Train the classifier with the given dataset.
        """
        logger.info("Training the classifier")
        self.classifier.fit(X_train, y_train)

    def evaluate(self, X_test, y_test):
        """
        Evaluate the classifier performance on the test dataset.
        """
        logger.info("Evaluating the model performance")
        y_pred = self.classifier.predict(X_test)
        return classification_report(y_test, y_pred)

    def predict(self, text):
        """
        Predict the domain of a given text.
        """
        logger.info("Predicting the domain for given text")
        vectorized_text = self.vectorizer.transform([text])
        return self.classifier.predict(vectorized_text)[0]

    def save_model(self, output_path):
        """
        Save the trained model to a file.
        """
        logger.info(f"Saving the trained model to: {output_path}")
        with open(output_path, 'wb') as f:
            pickle.dump({
                'vectorizer': self.vectorizer,
                'classifier': self.classifier
            }, f)

    @classmethod
    def load_model(cls, model_path):
        """
        Load a trained model from a file.
        """
        logger.info(f"Loading model from: {model_path}")
        with open(model_path, 'rb') as f:
            model_data = pickle.load(f)

        classifier = cls()
        classifier.vectorizer = model_data['vectorizer']
        classifier.classifier = model_data['classifier']
        return classifier


# Example usage
if __name__ == "__main__":
    # Initialize classifier with the JSON file containing domain keywords
    classifier = ResearchDomainClassifier(domain_keywords_file='domain_keywords.json')

    # Prepare dataset
    pdf_directory = 'D:\\research_summarizer\\data\\pdf_files'
    X_train, X_test, y_train, y_test = classifier.prepare_dataset(pdf_directory)

    # Train model
    classifier.train(X_train, y_train)

    # Evaluate performance
    logger.info("Model evaluation:")
    print(classifier.evaluate(X_test, y_test))

    # Save model
    classifier.save_model('D:\\research_summarizer\\models\\domain_classifier.pkl')

    # Load and use model
    loaded_classifier = ResearchDomainClassifier.load_model('D:\\research_summarizer\\models\\domain_classifier.pkl')
    sample_text = "Neural networks are a core concept in machine learning."
    logger.info(f"Predicted Domain for sample text: {sample_text}")
    print(f"Predicted Domain: {loaded_classifier.predict(sample_text)}")
