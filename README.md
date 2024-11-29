# Brieflet: Research Paper Summarization and Analysis

Brieflet is a powerful tool that streamlines the process of understanding and extracting insights from research papers. It combines machine learning models to automate the classification of research domains and generation of paper summaries.

## Key Features

1. **Domain Classification**: Brieflet utilizes a pre-trained domain classification model to identify the subject area of a research paper based on its title and abstract. This helps you quickly understand the field of study.

2. **Summarization**: The tool employs advanced text summarization techniques to generate concise, yet informative summaries of research papers. This allows you to quickly grasp the key points without reading the entire document.

3. **Key Sections Extraction**: Brieflet extracts and presents the crucial sections of a research paper, including the abstract, introduction, methodology, results, and conclusion. This provides a structured overview of the paper's content.

4. **PDF Support**: Users can simply upload their PDF research papers, and Brieflet will process the content and provide the full analysis, including domain classification, key sections, and summary.

5. **Intuitive API**: The project exposes a user-friendly API that allows integration with other applications or custom workflows. Developers can leverage Brieflet's capabilities to enhance their own research-related projects.

## Installation and Setup

1. **Clone the Repository**:
   ```
   git clone https://github.com/your-username/brieflet.git
   ```

2. **Install Dependencies**:
   ```
   cd brieflet
   pip install -r requirements.txt
   ```
   This will install the required Python packages, including PyTorch, Transformers, pandas, scikit-learn, NLTK, and PyPDF2.

3. **Set Up NLTK**:
   NLTK requires some additional resources for tokenization, stopword removal, and other NLP tasks. Run the following Python script to download the required NLTK data:
   ```python
   import nltk
   nltk.download('punkt')  # Tokenizer models
   nltk.download('stopwords')  # Stopword lists
   ```

4. **Prepare the Environment**:
   - Create a directory to store the uploaded PDF files:
     ```
     mkdir -p data/pdf_files
     ```
   - Ensure the pre-trained domain classification and summarization models are available in the `models/` directory. If not, you can download and place them in the appropriate location.

## Usage

1. **Run the Flask API**:
   ```
   python api.py
   ```
   This will start the Brieflet API server on `http://localhost:5000`.

2. **Analyze a Research Paper**:
   You can interact with the API using a tool like Postman or a simple HTML form. Here's an example using cURL:
   ```
   curl -X POST \
     -F 'file=@/path/to/your/research_paper.pdf' \
     http://localhost:5000/analyze
   ```
   Replace `/path/to/your/research_paper.pdf` with the actual path to your PDF file.

   The API will process the uploaded PDF and return a JSON response with the following information:
   - `status`: Indicates whether the processing was successful or not.
   - `filename`: The name of the uploaded file.
   - `domain`: The classified domain or subject area of the research paper.
   - `sections`: A dictionary containing the extracted key sections of the paper, including the abstract, introduction, methodology, results, and conclusion.
   - `summary`: A summary of the research paper, including the original length and the summary length.

3. **Check the Health of the Models**:
   You can use the `/health` endpoint to check the status of the domain classifier and summarizer models:
   ```
   curl http://localhost:5000/health
   ```
   This will return a JSON response indicating whether the models have been loaded successfully.

## File Structure

```
brieflet/
├── data/
│   └── pdf_files/
├── models/
│   ├── domain_classifier.py
│   ├── domain_keywords.json
│   └── domain.log
├── src/
│   ├── logs/
│   └── _init_.py
|   └── summary.py
├── api.py
├── requirements.txt
└── README.md
```

- `data/pdf_files`: This directory is used to store the uploaded PDF research papers.
- `models/`: This directory contains the pre-trained domain classification and summarization models, as well as the domain keywords JSON file.
- `src/`:
  - `domain_classifier.py`: This module contains the implementation of the domain classification functionality.
  - `summary.py`: This module handles the research paper summarization process.
- `api.py`: The main entry point for the Flask API.
- `requirements.txt`: The list of required dependencies for the project.
- `README.md`: The project's documentation file you're currently reading.

## Contributing

If you find any issues or have suggestions for improvements, feel free to open an issue or submit a pull request. Contributions are always welcome!

## License

This project is licensed under the [MIT License](LICENSE).
