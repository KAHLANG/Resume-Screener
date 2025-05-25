# utils/docx_parser.py
import docx

def parse_docx(file_path):
    """
    Parses a DOCX file and extracts its text content.

    Args:
        file_path (str): The path to the DOCX file.

    Returns:
        str: The extracted text from the DOCX, or an empty string if an error occurs.
    """
    text = ""
    try:
        doc = docx.Document(file_path)
        for para in doc.paragraphs:
            text += para.text + "\n"
    except Exception as e:
        # Print an error message if parsing fails
        print(f"Error parsing DOCX {file_path}: {e}")
        text = "" # Return empty string on error
    return text
