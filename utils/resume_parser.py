# utils/resume_parser.py
import spacy
import re
from .text_preprocessing import preprocess_text, extract_skills_spacy, extract_years_of_experience, nlp # Import necessary functions and nlp object

def parse_resume(text):
    """
    Parses raw resume text to extract key information like name, processed text,
    skills, and years of experience.

    Args:
        text (str): The raw text content of a resume.

    Returns:
        dict: A dictionary containing extracted resume information.
    """
    processed_text = preprocess_text(text)
    doc = nlp(text) # Use original text for NER for better name detection

    # Simple NER for name: Look for PERSON entities, prioritize longer ones or those at the beginning
    name = ""
    for ent in doc.ents:
        if ent.label_ == "PERSON" and len(ent.text.split()) > 1:
            # Heuristic: often the first PERSON entity or a prominent one is the name
            name = ent.text.title() # Capitalize first letter of each word
            break # Take the first reasonable name found

    # Extract skills using the shared utility function
    skills = extract_skills_spacy(processed_text)

    # Extract years of experience using the shared utility function
    experience = extract_years_of_experience(text) # Use original text for experience regex

    return {
        "name": name,
        "processed_text": processed_text,
        "skills": skills,
        "years_experience": experience
    }

def parse_job_description(text):
    """
    Parses raw job description text to extract job title, processed text,
    required skills, and required years of experience.

    Args:
        text (str): The raw text content of a job description.

    Returns:
        dict: A dictionary containing extracted job description information.
    """
    processed_text = preprocess_text(text)
    doc = nlp(text) # Use original text for NER/title detection

    # Simple approach to extract job title:
    # Look for capitalized phrases at the beginning of the text or common job title patterns.
    # This is a heuristic and can be improved with more advanced NLP or a list of common job titles.
    job_title = "N/A"
    # Attempt to find a prominent noun phrase near the start
    for token in doc[:20]: # Look at the first 20 tokens
        if token.pos_ == "NOUN" and token.is_title and len(token.text) > 2:
            job_title = token.text
            break
    # Fallback to a simple regex for common job title patterns
    if job_title == "N/A":
        title_match = re.search(r'(?:(?:senior|junior|lead|staff|principal)\s+)?(?:data\s+scientist|software\s+engineer|machine\s+learning\s+engineer|analyst|developer|manager|specialist|architect|consultant)', text, re.IGNORECASE)
        if title_match:
            job_title = title_match.group(0).title()


    # Extract required skills using the shared utility function
    required_skills = extract_skills_spacy(processed_text)

    # Extract required years of experience using the shared utility function
    required_experience = extract_years_of_experience(text) # Use original text for experience regex

    # Placeholder for domain-specific keywords - can be expanded
    domain_keywords = []

    return {
        "job_title": job_title,
        "processed_text": processed_text,
        "required_skills": required_skills,
        "required_experience": required_experience,
        "domain_keywords": domain_keywords
    }
