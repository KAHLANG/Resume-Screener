# utils/pdf_parser.py
import fitz  # PyMuPDF

def parse_pdf(file_path):
    """
    Parses a PDF file and extracts its text content.

    Args:
        file_path (str): The path to the PDF file.

    Returns:
        str: The extracted text from the PDF, or an empty string if an error occurs.
    """
    text = ""
    try:
        doc = fitz.open(file_path)
        for page in doc:
            text += page.get_text()
        doc.close()
    except Exception as e:
        # Print an error message if parsing fails
        print(f"Error parsing PDF {file_path}: {e}")
        text = "" # Return empty string on error to avoid breaking the application
    return text
