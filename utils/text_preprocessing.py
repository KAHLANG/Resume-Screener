# utils/text_preprocessing.py
import spacy
import re
from nltk.corpus import stopwords
import string
import nltk

# Load spaCy model
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    # If model not found, download it
    print("Downloading spaCy model 'en_core_web_sm'...")
    spacy.cli.download("en_core_web_sm")
    nlp = spacy.load("en_core_web_sm")

# NLTK stopwords (for better coverage and robustness)
try:
    nltk.data.find('corpora/stopwords')
except nltk.downloader.DownloadError:
    # Download stopwords if not already present
    print("Downloading NLTK stopwords...")
    nltk.download('stopwords')
finally:
    stopwords_set = set(stopwords.words('english'))

def preprocess_text(text):
    """
    Performs standard text preprocessing steps:
    - Lowercasing
    - Punctuation removal
    - Number removal
    - Extra whitespace removal
    - Lemmatization
    - Stopword removal

    Args:
        text (str): The input text to preprocess.

    Returns:
        str: The cleaned and preprocessed text.
    """
    # Convert text to lowercase
    text = text.lower()
    # Remove punctuation using string.punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))
    # Remove numbers (digits)
    text = re.sub(r'\d+', '', text)
    # Replace multiple spaces with a single space and strip leading/trailing whitespace
    text = re.sub(r'\s+', ' ', text).strip()

    # Process text with spaCy for lemmatization and tokenization
    doc = nlp(text)
    # Lemmatize tokens and remove stopwords
    tokens = [token.lemma_ for token in doc if token.text not in stopwords_set]
    return " ".join(tokens)

def extract_skills_spacy(text):
    """
    Extracts potential skills from text using a combination of keyword matching
    and spaCy's Named Entity Recognition (NER).

    Args:
        text (str): The preprocessed text (e.g., from a resume or job description).

    Returns:
        list: A list of unique skills identified.
    """
    doc = nlp(text)
    skills = set() # Use a set to automatically handle duplicates

    # Common tech skills keywords (can be expanded based on common job roles and industries)
    tech_keywords = set([
        "python", "java", "sql", "aws", "azure", "gcp", "docker", "kubernetes",
        "machine learning", "deep learning", "nlp", "data analysis",
        "tableau", "power bi", "excel", "javascript", "react", "angular",
        "node.js", "frontend", "backend", "fullstack", "devops", "cloud",
        "agile", "scrum", "git", "linux", "api", "rest", "graphql", "tensorflow",
        "pytorch", "scikit-learn", "spark", "hadoop", "etl", "data warehousing",
        "business intelligence", "statistical analysis", "data visualization",
        "c++", "c#", "php", "ruby", "golang", "swift", "kotlin", "android", "ios",
        "cybersecurity", "network security", "cloud security", "ethical hacking",
        "penetration testing", "vulnerability assessment", "incident response",
        "blockchain", "cryptography", "smart contracts", "web development",
        "ui/ux", "product management", "project management", "jira", "confluence",
        "salesforce", "sap", "erp", "crm", "data science", "big data", "r",
        "matlab", "sas", "ssis", "ssrs", "ssas", "bash", "shell scripting",
        "automation", "robotics", "iot", "computer vision", "time series",
        "predictive modeling", "statistical modeling", "a/b testing",
        "data mining", "feature engineering", "model deployment", "mlobs",
        "communication", "teamwork", "leadership", "problem-solving", "critical thinking",
        "adaptability", "creativity", "time management", "attention to detail",
        "customer service", "sales", "marketing", "finance", "accounting", "hr",
        "supply chain", "logistics", "operations", "research", "writing", "editing",
        "public speaking", "negotiation", "strategy", "consulting", "analytics",
        "statistics", "mathematics", "physics", "chemistry", "biology", "engineering",
        "design", "autocad", "solidworks", "photoshop", "illustrator", "figma", "sketch"
    ])

    # Simple keyword matching: check if any tech keyword is present in the text tokens
    text_tokens = set(text.split())
    for keyword in tech_keywords:
        if keyword in text_tokens or any(kw_part in text_tokens for kw_part in keyword.split() if len(kw_part) > 1):
            skills.add(keyword)
            
    # Look for capitalized words that might be skills (e.g., specific software names, frameworks)
    # This is applied to the original (non-lowercased) text for better detection of proper nouns
    original_doc = nlp(text) # Re-process original text for proper noun detection
    for token in original_doc:
        if token.is_title and len(token.text) > 1 and token.text.lower() not in stopwords_set:
            skills.add(token.text.lower()) # Add as lowercase to maintain consistency

    # Use spaCy's NER for entities that might represent skills (e.g., organizations, products, miscellaneous)
    # This is applied to the original (non-lowercased) text
    for ent in original_doc.ents:
        # Filter by entity labels commonly associated with skills/technologies
        if ent.label_ in ["ORG", "PRODUCT", "MISC", "GPE"] and len(ent.text.split()) < 5: # Limit length to avoid long phrases
            skills.add(ent.text.lower())
    
    # Refine skills: remove very short words or common non-skill words that might have slipped through
    # Ensure skills are not just single letters or common stopwords
    skills_filtered = [s for s in list(skills) if len(s) > 1 and s not in stopwords_set]
    return list(set(skills_filtered)) # Convert back to list and ensure uniqueness

def extract_years_of_experience(text):
    """
    Extracts years of experience from text using regular expressions.
    Looks for patterns like "X years", "X+ years", "X-Y years", "minimum X years", "X yrs".

    Args:
        text (str): The input text (e.g., from a resume or job description).

    Returns:
        int: The maximum number of years of experience found, or 0 if none.
    """
    # Regex to capture various forms of experience mentions
    matches = re.findall(
        r'(\d+)\s*(?:-|to)?\s*(\d+)?\s*(?:year|yr)s?(?:\s*of)?\s*(?:relevant|minimum|total|professional)?\s*experience|\b(\d+)\+\s*(?:year|yr)s?\b',
        text, re.IGNORECASE
    )
    years = []
    for match in matches:
        if match[0]: # Matches "X years" or "X-Y years"
            years.append(int(match[0]))
            if match[1]: # If Y is present in "X-Y years"
                years.append(int(match[1]))
        elif match[2]: # Matches "X+ years"
            years.append(int(match[2]))
    
    # Return the maximum years found, or 0 if no experience pattern is matched
    return max(years) if years else 0
