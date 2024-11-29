import os
import pandas as pd
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report
import PyPDF2


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
        # Core Domains
        "Artificial Intelligence": [
            "AI", "machine learning", "deep learning", "neural network", "natural language processing",
            "robotics", "computer vision", "chatbots", "autonomous systems", "reinforcement learning",
            "generative models", "decision trees", "intelligent systems"
        ],
        "Biology": [
            "DNA", "genome", "biochemistry", "microbiology", "protein", "cell", "neuroscience",
            "evolution", "genetics", "molecular biology", "enzymes", "bioinformatics",
            "zoology", "botany", "ecology", "virology", "taxonomy", "biosynthesis"
        ],
        "Physics": [
            "quantum mechanics", "relativity", "thermodynamics", "particle physics", "optics", "cosmology",
            "fluid dynamics", "electromagnetism", "nuclear physics", "astrophysics", "string theory",
            "quantum computing", "condensed matter physics", "plasma physics", "wave theory"
        ],
        "Chemistry": [
            "chemical reaction", "organic chemistry", "inorganic chemistry", "biochemistry", "molecules",
            "polymers", "nanotechnology", "spectroscopy", "catalysts", "analytical chemistry",
            "physical chemistry", "pharmacology", "electrochemistry", "material chemistry"
        ],
        "Mathematics": [
            "calculus", "algebra", "geometry", "statistics", "topology", "theorems",
            "number theory", "probability", "differential equations", "graph theory", "linear algebra",
            "cryptography", "mathematical modeling", "game theory", "optimization", "combinatorics"
        ],
        "Computer Science": [
            "programming", "algorithm", "data structure", "cybersecurity", "cloud computing",
            "databases", "blockchain", "distributed systems", "networking", "computational theory",
            "software engineering", "web development", "artificial intelligence ethics"
        ],
        "Medicine": [
            "pharmacology", "anatomy", "pathology", "epidemiology", "oncology", "cardiology",
            "immunology", "neurology", "surgery", "psychiatry", "virology", "genomics",
            "dermatology", "pediatrics", "orthopedics", "radiology", "emergency medicine"
        ],
        "Engineering": [
            "mechanical engineering", "civil engineering", "electrical engineering", "robotics",
            "nanotechnology", "aerospace engineering", "biomedical engineering", "energy systems",
            "automation", "structural analysis", "systems engineering", "control systems"
        ],
        "Environmental Science": [
            "climate change", "biodiversity", "conservation", "pollution", "sustainability",
            "ecosystems", "carbon footprint", "environmental monitoring", "renewable energy", "ecology",
            "water resources", "waste management", "environmental ethics", "geospatial analysis"
        ],
        "Economics": [
            "macroeconomics", "microeconomics", "game theory", "trade", "finance", "economy",
            "investment", "market analysis", "fiscal policy", "monetary policy", "economic growth",
            "development economics", "behavioral economics", "public policy", "econometrics"
        ],
        "History": [
            "ancient history", "modern history", "archaeology", "war", "historical analysis",
            "colonialism", "revolutions", "medieval history", "cultural heritage", "historical artifacts",
            "world wars", "industrial revolution", "historical texts", "dynasties"
        ],

        # Interdisciplinary Domains
        "Social Sciences": [
            "sociology", "psychology", "anthropology", "political science", "demographics",
            "social behavior", "cultural studies", "human geography", "gender studies",
            "criminology", "public administration", "international relations"
        ],
        "Education": [
            "pedagogy", "learning theories", "e-learning", "curriculum development", "educational psychology",
            "teacher training", "special education", "assessment", "educational policy",
            "distance learning", "educational technology", "student engagement", "learning analytics"
        ],
        "Linguistics": [
            "phonetics", "syntax", "semantics", "pragmatics", "language acquisition",
            "sociolinguistics", "historical linguistics", "computational linguistics", "language modeling",
            "multilingualism", "morphology", "dialects", "translation studies"
        ],
        "Philosophy": [
            "ethics", "metaphysics", "epistemology", "logic", "aesthetics",
            "existentialism", "moral philosophy", "philosophy of science", "philosophy of mind",
            "phenomenology", "philosophy of language", "stoicism", "utilitarianism"
        ],
        "Law": [
            "constitutional law", "criminal law", "civil law", "intellectual property", "human rights",
            "international law", "legal ethics", "environmental law", "corporate law", "cyber law",
            "contract law", "family law", "legal jurisprudence", "privacy laws"
        ],
        "Business and Management": [
            "marketing", "human resources", "business strategy", "entrepreneurship", "supply chain management",
            "organizational behavior", "accounting", "finance", "consumer behavior", "operations management",
            "risk management", "leadership", "corporate governance", "business analytics"
        ],
        "Astronomy": [
            "stars", "planets", "galaxies", "black holes", "space exploration",
            "telescope", "astronomical observations", "astrochemistry", "astrobiology", "celestial mechanics",
            "space weather", "exoplanets", "cosmic microwave background", "gravitational waves"
        ],
        "Geology": [
            "earthquakes", "volcanology", "mineralogy", "sedimentology", "petrology",
            "plate tectonics", "geochronology", "paleontology", "geomorphology", "geophysics",
            "seismology", "stratigraphy", "hydrogeology", "geothermal energy"
        ],
        "Arts and Humanities": [
            "literature", "visual arts", "performing arts", "art history", "philosophy of art",
            "creative writing", "cultural studies", "aesthetics", "musicology", "theater studies",
            "film studies", "dance", "design theory", "folk art"
        ],
        "Materials Science": [
            "nanomaterials", "composites", "metallurgy", "polymers", "ceramics",
            "crystallography", "thin films", "materials engineering", "semiconductors", "alloys",
            "biomaterials", "smart materials", "material synthesis", "surface science"
        ],
        "Statistics and Data Science": [
            "data analysis", "probability", "machine learning", "big data", "data visualization",
            "time series", "Bayesian statistics", "statistical modeling", "data mining", "predictive analytics",
            "data engineering", "decision trees", "feature engineering", "unsupervised learning"
        ],
        "Energy": [
            "renewable energy", "solar power", "wind power", "fossil fuels", "nuclear energy",
            "energy policy", "energy storage", "battery technology", "energy efficiency", "power grids",
            "hydropower", "tidal energy", "biofuels", "geothermal energy"
        ],
        "Transportation": [
            "logistics", "traffic management", "public transportation", "autonomous vehicles", "aviation",
            "railways", "shipping", "transport engineering", "infrastructure", "urban planning",
            "smart cities", "hyperloop", "mobility solutions", "electric vehicles"
        ],
        "Agriculture": [
            "crop science", "soil science", "agronomy", "horticulture", "sustainable agriculture",
            "irrigation", "food security", "plant pathology", "livestock management", "agricultural economics",
            "precision farming", "pesticides", "organic farming", "agricultural technology"
        ],
        "Psychology": [
            "cognitive psychology", "behavioral psychology", "clinical psychology", "developmental psychology",
            "neuropsychology", "psychotherapy", "mental health", "personality disorders", "social psychology",
            "psychometrics", "addiction psychology", "educational psychology", "positive psychology"
        ]
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
        self.classifier.fit(X_train, y_train)

    def evaluate(self, X_test, y_test):
        y_pred = self.classifier.predict(X_test)
        return classification_report(y_test, y_pred)

    def predict(self, text):
        vectorized_text = self.vectorizer.transform([text])
        return self.classifier.predict(vectorized_text)[0]

    def save_model(self, output_path):
        with open(output_path, 'wb') as f:
            pickle.dump({
                'vectorizer': self.vectorizer,
                'classifier': self.classifier
            }, f)

    @classmethod
    def load_model(cls, model_path):
        with open(model_path, 'rb') as f:
            model_data = pickle.load(f)

        classifier = cls()
        classifier.vectorizer = model_data['vectorizer']
        classifier.classifier = model_data['classifier']
        return classifier


# Example usage
if __name__ == "__main__":
    # Initialize classifier
    classifier = ResearchDomainClassifier()

    # Prepare dataset
    pdf_directory = 'D:\\research_summarizer\\data\\pdf_files'
    X_train, X_test, y_train, y_test = classifier.prepare_dataset(pdf_directory)

    # Train model
    classifier.train(X_train, y_train)

    # Evaluate performance
    print(classifier.evaluate(X_test, y_test))

    # Save model
    classifier.save_model('D:\\research_summarizer\\models\\domain_classifier.pkl')

    # Load and use model
    loaded_classifier = ResearchDomainClassifier.load_model('D:\\research_summarizer\\models\\domain_classifier.pkl')
    sample_text = "Neural networks are a core concept in machine learning."
    print(f"Predicted Domain: {loaded_classifier.predict(sample_text)}")
