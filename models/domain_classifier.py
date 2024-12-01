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
            "Artificial Intelligence": ["AI", "machine learning", "deep learning", "neural network", "natural language processing", "reinforcement learning", "computer vision", "chatbot", "pattern recognition"],
            "Biology": ["DNA", "genome", "biochemistry", "microbiology", "protein", "genetics", "ecology", "evolution", "immunology", "cell biology", "physiology", "biophysics"],
            "Physics": ["quantum mechanics", "thermodynamics", "particle physics", "optics", "astrophysics", "electromagnetism", "nuclear physics", "mechanics", "fluid dynamics", "relativity", "condensed matter"],
            "Chemistry": ["chemical reaction", "organic chemistry", "biochemistry", "inorganic chemistry", "analytical chemistry", "molecular biology", "stoichiometry", "thermochemistry", "reaction kinetics", "materials science"],
            "Mathematics": ["calculus", "algebra", "geometry", "statistics", "probability", "differential equations", "number theory", "linear algebra", "discrete mathematics", "topology", "mathematical modeling"],
            "Computer Science": ["programming", "algorithm", "data structure", "cybersecurity", "databases", "software engineering", "networking", "cloud computing", "big data", "artificial intelligence", "machine learning", "operating systems", "computational theory", "data science"],
            "Engineering": ["mechanical engineering", "civil engineering", "electrical engineering", "chemical engineering", "aerospace engineering", "structural engineering", "electromagnetics", "control systems", "robotics", "manufacturing", "industrial engineering"],
            "Economics": ["microeconomics", "macroeconomics", "supply and demand", "market equilibrium", "econometrics", "game theory", "international trade", "inflation", "finance", "economic policy", "monetary policy", "economic growth"],
            "Psychology": ["behaviorism", "cognitive psychology", "neuropsychology", "social psychology", "developmental psychology", "clinical psychology", "personality psychology", "psychopathology", "research methodology", "psychotherapy", "counseling"],
            "History": ["ancient history", "medieval history", "modern history", "world wars", "historical events", "archaeology", "historical research", "cultural history", "historical analysis", "political history"],
            "Literature": ["poetry", "novels", "short stories", "drama", "literary analysis", "fiction", "non-fiction", "literary criticism", "classical literature", "modernism", "postmodernism", "mythology"],
            "Philosophy": ["ethics", "metaphysics", "epistemology", "logic", "philosophy of mind", "philosophy of science", "existentialism", "utilitarianism", "aesthetics", "morality", "political philosophy", "metaphysical realism"],
            "Medicine": ["anatomy", "pathology", "pharmacology", "neurology", "cardiology", "oncology", "immunology", "surgery", "psychiatry", "clinical research", "epidemiology", "infectious diseases", "preventive medicine", "pediatrics"],
            "Art": ["painting", "sculpture", "modern art", "renaissance art", "art history", "photography", "visual arts", "art theory", "aesthetic", "artistic movements", "abstract art", "street art", "installation art"],
            "Sociology": ["social structure", "culture", "society", "social interaction", "urban sociology", "deviance", "social norms", "social psychology", "class", "race and ethnicity", "sociological theory", "globalization", "inequality"],
            "Environmental Science": ["ecosystem", "climate change", "biodiversity", "conservation", "pollution", "renewable energy", "sustainability", "ecology", "environmental policy", "natural resources", "environmental impact", "waste management"],
            "Linguistics": ["phonetics", "syntax", "semantics", "morphology", "sociolinguistics", "psycholinguistics", "language acquisition", "phonology", "linguistic theory", "applied linguistics", "language typology", "pragmatics"],
            "Political Science": ["government", "political theory", "political systems", "public policy", "international relations", "comparative politics", "political parties", "political ideologies", "globalization", "democracy", "political economy"],
            "Geography": ["physical geography", "human geography", "cartography", "geomorphology", "climate", "urban planning", "geopolitics", "topography", "demographics", "geographical information systems", "spatial analysis"],
            "Law": ["constitutional law", "criminal law", "civil law", "international law", "human rights", "legal theory", "law enforcement", "jurisprudence", "contracts", "litigation", "intellectual property", "administrative law"],
            "Business": ["management", "marketing", "finance", "entrepreneurship", "strategy", "operations", "human resources", "organizational behavior", "international business", "leadership", "supply chain management", "business ethics"],
            "Education": ["curriculum development", "teaching methods", "pedagogy", "educational psychology", "school administration", "distance learning", "educational technology", "special education", "higher education", "learning outcomes"],
            "Mathematical Finance": ["derivatives", "options pricing", "financial modeling", "portfolio theory", "risk management", "quantitative analysis", "stochastic processes", "financial instruments", "capital markets"],
            "Anthropology": ["cultural anthropology", "archaeology", "linguistic anthropology", "physical anthropology", "ethnography", "human evolution", "social structure", "ethnology", "primatology"],
            "Theology": ["Christianity", "Islam", "Judaism", "Buddhism", "philosophy of religion", "sacred texts", "religious studies", "spirituality", "ethics", "comparative religion"],
            "Statistics": ["probability", "statistical inference", "hypothesis testing", "regression analysis", "data analysis", "Bayesian statistics", "sampling", "statistical modeling", "time series analysis", "biostatistics"],
            "Agriculture": ["agronomy", "crop science", "soil science", "agricultural economics", "plant breeding", "sustainable agriculture", "agriculture technology", "horticulture", "animal science", "agroforestry"],
            "Veterinary Science": ["animal health", "veterinary medicine", "zoology", "veterinary pathology", "disease prevention", "animal nutrition", "veterinary surgery", "veterinary diagnostics", "wildlife conservation"],
            "Music": ["composition", "music theory", "performance", "musical genres", "ethnomusicology", "music history", "conducting", "musical instruments", "orchestration", "audio engineering"],
            "Architecture": ["urban planning", "building design", "construction", "sustainable architecture", "landscape architecture", "interior design", "architectural theory", "historic preservation", "architectural engineering", "city design"],
            "Tourism": ["hospitality", "tourism management", "tourism marketing", "tourism economics", "cultural tourism", "eco-tourism", "tourist behavior", "destination management", "travel industry"],
            "Sports Science": ["exercise physiology", "sports psychology", "kinesiology", "athletic training", "sports medicine", "biomechanics", "sports nutrition", "fitness", "sports performance", "sports injury prevention"],
            "Cognitive Science": ["cognitive psychology", "neuroscience", "artificial intelligence", "decision-making", "learning", "memory", "perception", "language", "brain-computer interfaces", "neuroplasticity"],
            "Cryptocurrency": ["blockchain", "bitcoin", "ethereum", "cryptographic security", "digital currency", "distributed ledger", "cryptography", "decentralization", "smart contracts", "mining", "tokenomics"],
            "Social Work": ["child welfare", "social justice", "community development", "mental health", "family therapy", "counseling", "clinical social work", "social policy", "advocacy", "public health"],
            "Fashion": ["design", "textiles", "fashion trends", "fashion marketing", "apparel", "costume design", "fashion history", "sustainability", "fashion entrepreneurship", "modeling"],
            "Film Studies": ["cinema", "film history", "directing", "film theory", "screenwriting", "cinematography", "editing", "sound design", "documentary", "film criticism"]
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
