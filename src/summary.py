"""
A module to process research papers and extract summaries and key sections.
"""
import pickle
import logging
import re
import os
import PyPDF2
import nltk
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM


# Configure logging
LOG_DIR = 'logs'
os.makedirs(LOG_DIR, exist_ok=True)

logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f"{LOG_DIR}/summarizer.log", mode='a', encoding='utf-8'),
        logging.StreamHandler()
    ]
)


class ResearchPaperProcessor:
    """
    A class to process research papers and extract summaries and key sections.
    """

    def __init__(self, pdf_path):
        """
        Initialize the ResearchPaperProcessor.

        Args:
            pdf_path (str): Path to the research paper PDF.
        """
        self.pdf_path = pdf_path
        self.tokenizer = AutoTokenizer.from_pretrained('facebook/bart-large-cnn')
        self.summary_model = AutoModelForSeq2SeqLM.from_pretrained(
            'facebook/bart-large-cnn'
        )
        logging.info("ResearchPaperProcessor initialized.")

    def extract_text_from_pdf(self):
        """
        Extract text from the provided PDF.

        Returns:
            str: Extracted text from the PDF.

        Raises:
            Exception: If PDF extraction fails.
        """
        try:
            with open(self.pdf_path, 'rb') as file:
                reader = PyPDF2.PdfReader(file)
                full_text = ""

                for page in reader.pages:
                    full_text += page.extract_text() + "\n"

                logging.info("Successfully extracted text from %s", self.pdf_path)
                return full_text
        except Exception as error:
            logging.error("PDF extraction error: %s", error, exc_info=True)
            raise

    @staticmethod
    def preprocess_text(text):
        """
        Preprocess the extracted text.

        Args:
            text (str): Raw text to preprocess.

        Returns:
            str: Cleaned and filtered text.
        """
        text = re.sub(r'\s+', ' ', text).strip()
        text = re.sub(r'\d+', '', text)
        sentences = nltk.sent_tokenize(text)
        filtered_sentences = [
            sent for sent in sentences if 10 < len(sent.split()) < 50
        ]

        return ' '.join(filtered_sentences)

    @staticmethod
    def extract_key_sections(text):
        """
        Extract key sections from the text.

        Args:
            text (str): Preprocessed text.

        Returns:
            dict: Dictionary containing extracted sections.
        """
        sections = {
            'abstract': re.search(
                r'abstract.*?(?=introduction)', text,
                re.IGNORECASE | re.DOTALL
            ),
            'introduction': re.search(
                r'introduction.*?(?=method|methodology)', text,
                re.IGNORECASE | re.DOTALL
            ),
            'methodology': re.search(
                r'method.*?(?=result)', text,
                re.IGNORECASE | re.DOTALL
            ),
            'results': re.search(
                r'result.*?(?=conclusion)', text,
                re.IGNORECASE | re.DOTALL
            ),
            'conclusion': re.search(
                r'conclusion.*', text,
                re.IGNORECASE | re.DOTALL
            )
        }
        return {k: v.group(0) if v else '' for k, v in sections.items()}

    def generate_summary(self, text, max_length=250):
        """
        Generate a summary of the provided text.

        Args:
            text (str): Text to summarize.
            max_length (int, optional): Maximum length of the summary. Defaults to 250.

        Returns:
            dict: Dictionary containing the summary and text lengths.
        """
        inputs = self.tokenizer(
            text, max_length=1024, truncation=True, return_tensors='pt'
        )

        summary_ids = self.summary_model.generate(
            inputs['input_ids'], max_length=max_length,
            num_beams=4, early_stopping=True
        )

        summary = self.tokenizer.decode(summary_ids[0], skip_special_tokens=True)
        logging.debug("Generated summary successfully.")

        return {
            'summary': summary,
            'original_length': len(text.split()),
            'summary_length': len(summary.split())
        }

    def save_to_pickle(self, filename="summarizer.pkl"):
        """
        Save the tokenizer and summarization model to a pickle file.

        Args:
            filename (str, optional): File name to save the models. Defaults to "summarizer.pkl".
        """
        try:
            with open(filename, 'wb') as f:
                pickle.dump({
                    'tokenizer': self.tokenizer,
                    'summary_model': self.summary_model
                }, f)
            logging.info(f"Model and tokenizer saved to {filename}")
        except Exception as e:
            logging.error(f"Failed to save models: {e}")
            raise


def main(pdf_path):
    """
    Main function to process the research paper.

    Args:
        pdf_path (str): Path to the PDF file.

    Returns:
        dict: Dictionary containing the sections and summary.

    Raises:
        Exception: If processing fails.
    """
    try:
        processor = ResearchPaperProcessor(pdf_path)
        full_text = processor.extract_text_from_pdf()
        cleaned_text = processor.preprocess_text(full_text)
        sections = processor.extract_key_sections(cleaned_text)
        summary = processor.generate_summary(cleaned_text)
        processor.save_to_pickle()  # Save the model and tokenizer to a pickle file
        logging.info("Processing complete.")

        return {
            'sections': sections,
            'summary': summary
        }
    except Exception as error:
        logging.error("Processing failed: %s", error, exc_info=True)
        raise


if __name__ == "__main__":
    PDF_PATH = 'D:\\Brieflet\\data\\pdf_files\\Neural Networks from Scratch in Python.pdf'
    RESULTS = main(PDF_PATH)
    print(RESULTS)
