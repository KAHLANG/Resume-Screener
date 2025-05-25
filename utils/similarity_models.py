# utils/similarity_models.py
from sentence_transformers import SentenceTransformer, util
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import torch # Import torch for tensor operations

# Global variable to store the BERT model
bert_model = None

# Load BERT model once when the module is imported
# This prevents reloading the model multiple times in Streamlit, which can be slow
try:
    bert_model = SentenceTransformer('all-MiniLM-L6-v2')
    print("SentenceTransformer model 'all-MiniLM-L6-v2' loaded successfully.")
except Exception as e:
    print(f"Error loading SentenceTransformer model: {e}")
    print("Attempting to download 'all-MiniLM-L6-v2'...")
    try:
        # This might trigger download if not available
        bert_model = SentenceTransformer('all-MiniLM-L6-v2')
        print("SentenceTransformer model 'all-MiniLM-L6-v2' downloaded and loaded.")
    except Exception as e:
        print(f"Failed to download and load SentenceTransformer model: {e}")
        bert_model = None # Set to None if loading fails, so functions can handle it gracefully

def get_bert_embedding(text):
    """
    Generates a BERT embedding for the given text.

    Args:
        text (str): The input text.

    Returns:
        torch.Tensor or numpy.ndarray: The BERT embedding (a vector),
                                       or a zero vector if the model failed to load.
    """
    if bert_model is None:
        # Return a zero vector if the model couldn't be loaded, to prevent errors
        print("BERT model not loaded. Returning a zero vector.")
        # all-MiniLM-L6-v2 produces 384-dimensional embeddings
        return torch.zeros(384) if torch.cuda.is_available() else np.zeros(384)
    return bert_model.encode(text, convert_to_tensor=True)

def calculate_cosine_similarity_bert(text1_embedding, text2_embedding):
    """
    Calculates the cosine similarity between two BERT embeddings.

    Args:
        text1_embedding (torch.Tensor): The BERT embedding of the first text.
        text2_embedding (torch.Tensor): The BERT embedding of the second text.

    Returns:
        float: The cosine similarity score (0.0 to 1.0), or 0.0 if embeddings are invalid.
    """
    if text1_embedding is None or text2_embedding is None or text1_embedding.numel() == 0 or text2_embedding.numel() == 0:
        return 0.0 # Return 0.0 if embeddings are invalid or empty
    # Ensure embeddings are on the same device (CPU/GPU) if applicable
    if text1_embedding.device != text2_embedding.device:
        text2_embedding = text2_embedding.to(text1_embedding.device)
    return util.pytorch_cos_sim(text1_embedding, text2_embedding).item()

def calculate_tfidf_cosine_similarity(text1, text2):
    """
    Calculates the cosine similarity between two texts using TF-IDF vectors.

    Args:
        text1 (str): The first text.
        text2 (str): The second text.

    Returns:
        float: The cosine similarity score (0.0 to 1.0), or 0.0 if texts are empty.
    """
    vectorizer = TfidfVectorizer()
    try:
        # Fit and transform both texts to TF-IDF vectors
        vectors = vectorizer.fit_transform([text1, text2])
        # Calculate cosine similarity between the two vectors
        score = cosine_similarity(vectors[0:1], vectors[1:2])[0][0]
    except ValueError:
        # This error typically occurs if input texts are empty or contain only stopwords
        score = 0.0
    return score
