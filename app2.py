from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import pandas as pd
import re
import pickle
import os
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
# For profanity detection - install with: pip install better-profanity
from better_profanity import profanity

# Ensure NLTK resources are downloaded
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)

app = Flask(__name__)
CORS(app)  # This allows your webpage to make requests to this API

# Initialize profanity detector
profanity.load_censor_words()

# Initialize lemmatizer for word normalization
lemmatizer = WordNetLemmatizer()

# Create a custom stopwords list that KEEPS negation words
custom_stopwords = set(stopwords.words("english")) - {
    "no", "not", "nor", "neither", "never", "none", "nobody", "nowhere", 
    "nothing", "ain't", "isn't", "aren't", "wasn't", "weren't", "haven't", 
    "hasn't", "hadn't", "doesn't", "don't", "didn't", "won't", "wouldn't", 
    "shouldn't", "can't", "cannot", "couldn't"
}

def improved_preprocess_text(tweet):
    """
    Preprocesses tweet text by:
    - Removing URLs, mentions, and formatting hashtags
    - Handling contractions and negations
    - Lemmatizing words
    - Creating bigrams for negations (e.g., "not good" becomes "not_good")
    
    Args:
        tweet (str): Input tweet text
        
    Returns:
        str: Preprocessed tweet text
    """
    if not isinstance(tweet, str):
        return ""
        
    # Remove URLs, mentions, hashtags
    tweet = re.sub(r'http\S+|www\S+|https\S+', '', tweet, flags=re.MULTILINE)
    tweet = re.sub(r'@\w+', '', tweet)
    
    # Keep hashtag content but remove the # symbol
    tweet = re.sub(r'#(\w+)', r'\1', tweet)
    
    # Handle common contractions
    contractions = {
        "ain't": "is not", "aren't": "are not", "can't": "cannot", "couldn't": "could not",
        "didn't": "did not", "doesn't": "does not", "don't": "do not", "hadn't": "had not",
        "hasn't": "has not", "haven't": "have not", "he'd": "he would", "he'll": "he will",
        "he's": "he is", "i'd": "i would", "i'll": "i will", "i'm": "i am", "i've": "i have",
        "isn't": "is not", "it's": "it is", "let's": "let us", "mightn't": "might not",
        "mustn't": "must not", "shan't": "shall not", "she'd": "she would", "she'll": "she will",
        "she's": "she is", "shouldn't": "should not", "that's": "that is", "there's": "there is",
        "they'd": "they would", "they'll": "they will", "they're": "they are", "they've": "they have",
        "we'd": "we would", "we're": "we are", "we've": "we have", "weren't": "were not",
        "what'll": "what will", "what're": "what are", "what's": "what is", "what've": "what have",
        "where's": "where is", "who'd": "who would", "who'll": "who will", "who're": "who are",
        "who's": "who is", "who've": "who have", "won't": "will not", "wouldn't": "would not",
        "you'd": "you would", "you'll": "you will", "you're": "you are", "you've": "you have"
    }
    for contraction, expansion in contractions.items():
        tweet = tweet.replace(contraction, expansion)
    
    # Convert to lowercase
    tweet = tweet.lower()
    
    # Handle special negation patterns
    tweet = re.sub(r'(\w+)n\'t', r'\1 not', tweet)
    
    # Replace non-alphanumeric with space but keep apostrophes for possessives
    tweet = re.sub(r'[^a-zA-Z\']', ' ', tweet)
    tweet = re.sub(r'\s+', ' ', tweet)  # Remove extra spaces
    
    # Split into words
    words = tweet.split()
    
    # Create bigrams for negations (e.g., "not good" becomes "not_good")
    bigrams = []
    negation_words = {"no", "not", "never", "none"}
    i = 0
    while i < len(words) - 1:
        if words[i] in negation_words and i + 1 < len(words):
            bigrams.append(f"{words[i]}_{words[i+1]}")
            i += 2
        else:
            if words[i] not in custom_stopwords and len(words[i]) > 2:
                bigrams.append(lemmatizer.lemmatize(words[i]))
            i += 1
    
    # Add the last word if it wasn't part of a bigram
    if i == len(words) - 1 and words[i] not in custom_stopwords and len(words[i]) > 2:
        bigrams.append(lemmatizer.lemmatize(words[i]))
    
    # Join back into a string
    processed_tweet = " ".join(bigrams)
    return processed_tweet

def predict_sentiment(text, model, vectorizer, use_advanced=True):
    """
    Predicts sentiment of input text using the provided model and vectorizer.
    Handles special cases like negations and profanity.
    
    Args:
        text (str): Input tweet text
        model: Trained sentiment classification model
        vectorizer: TF-IDF vectorizer for feature extraction
        use_advanced (bool): Whether to use advanced heuristics for prediction
        
    Returns:
        tuple: (prediction (0=negative, 1=positive), confidence score, processed_text)
    """
    # Check for profanity first
    contains_profanity = profanity.contains_profanity(text)
    
    # Check for negation words
    negation_words = ["not", "no", "never", "n't", "isn't", "aren't", "wasn't", 
                      "weren't", "haven't", "hasn't", "hadn't", "doesn't", 
                      "don't", "didn't", "won't", "wouldn't", "couldn't", 
                      "shouldn't", "can't", "cannot"]
    contains_negation = any(neg in text.lower() for neg in negation_words)
    
    # If we detect profanity and advanced options are enabled, directly classify as negative
    if contains_profanity and use_advanced:
        return 0, 0.9, "Detected profanity"  # Return negative with high confidence
    
    # Preprocess the text
    processed_text = improved_preprocess_text(text)
    
    # Vectorize the text
    text_tfidf = vectorizer.transform([processed_text])
    
    # Get prediction and probability
    prediction = model.predict(text_tfidf)[0]
    probabilities = model.predict_proba(text_tfidf)[0]
    confidence = probabilities[1] if prediction == 1 else probabilities[0]
    
    # Apply negation logic if advanced options are enabled
    if use_advanced and contains_negation and prediction == 1 and confidence < 0.8:
        # Flip prediction for negative statements that aren't strongly positive
        return 0, 0.7, processed_text
    
    return prediction, confidence, processed_text

def load_model_and_vectorizer(model_path, vectorizer_path):
    """
    Loads the trained model and vectorizer from the specified paths.
    
    Args:
        model_path (str): Path to the trained model file
        vectorizer_path (str): Path to the vectorizer file
        
    Returns:
        tuple: (model, vectorizer) if successful, (None, None) otherwise
    """
    try:
        with open(model_path, 'rb') as file:
            model = pickle.load(file)
        
        with open(vectorizer_path, 'rb') as file:
            vectorizer = pickle.load(file)
        
        return model, vectorizer
    
    except Exception as e:
        print(f"Error loading model or vectorizer: {e}")
        return None, None

# Paths to model and vectorizer - update these paths to your model location
# Using a more flexible approach to find models in subdirectories
def find_model_files():
    base_dirs = [".", "model", "../model", "./model"]
    model_filename = "sentiment_model.pkl"
    vectorizer_filename = "tfidf_vectorizer.pkl"
    gb_model_filename = "gb_sentiment_model.pkl"
    
    for base_dir in base_dirs:
        model_path = os.path.join(base_dir, model_filename)
        vectorizer_path = os.path.join(base_dir, vectorizer_filename)
        gb_model_path = os.path.join(base_dir, gb_model_filename)
        
        if os.path.exists(model_path) and os.path.exists(vectorizer_path):
            return model_path, vectorizer_path, gb_model_path
    
    # Default paths if not found
    return "model/sentiment_model.pkl", "model/tfidf_vectorizer.pkl", "model/gb_sentiment_model.pkl"

model_path, vectorizer_path, gb_model_path = find_model_files()

# Load models and vectorizer at startup
model, vectorizer = None, None
gb_model = None

@app.before_request
def initialize():
    global model, vectorizer, gb_model
    # Only load if not already loaded
    if model is None or vectorizer is None:
        # Load model and vectorizer
        model, vectorizer = load_model_and_vectorizer(model_path, vectorizer_path)
        if model is None or vectorizer is None:
            print("Error: Failed to load primary model or vectorizer. Please check the file paths.")
        else:
            print("Successfully loaded primary model and vectorizer")
        
        # Try to load gradient boosting model as well
        try:
            with open(gb_model_path, 'rb') as file:
                gb_model = pickle.load(file)
            print("Successfully loaded gradient boosting model")
        except Exception as e:
            print(f"Warning: Failed to load gradient boosting model: {e}")

# Fallback prediction function in case model loading fails
def fallback_predict_sentiment(text):
    # Simple rule-based sentiment analysis as fallback
    positive_words = ['good', 'great', 'happy', 'excellent', 'love', 'nice', 'best', 'awesome', 
                     'wonderful', 'fantastic', 'amazing', 'enjoy', 'like', 'perfect']
    negative_words = ['bad', 'worst', 'hate', 'terrible', 'awful', 'sad', 'poor', 'disappointed',
                     'horrible', 'dislike', 'sucks', 'fail', 'failure', 'wrong', 'annoying']
    
    score = 0
    text_lower = text.lower()
    
    # Check for negation words
    negation_words = ["not", "no", "never", "n't", "isn't", "aren't", "wasn't", "weren't"]
    contains_negation = any(neg in text_lower for neg in negation_words)
    
    for word in positive_words:
        if word in text_lower:
            score += 1
    
    for word in negative_words:
        if word in text_lower:
            score -= 1
    
    # Invert score if negation is present
    if contains_negation:
        score = -score
    
    is_positive = score > 0
    confidence = min(0.6, abs(score) * 0.1 + 0.5)  # Scale confidence based on word count
    return 1 if is_positive else 0, confidence, "Used fallback prediction"

@app.route('/analyze', methods=['POST'])
def analyze():
    data = request.json
    if not data or 'text' not in data:
        return jsonify({'error': 'No text provided'}), 400
    
    text = data['text']
    use_advanced = data.get('advanced', True)  # Default to using advanced options
    use_gb_model = data.get('use_gb', False)  # Option to use gradient boosting model
    
    global model, vectorizer, gb_model
    
    # Try both models if available and average results if requested
    if use_gb_model and gb_model is not None and vectorizer is not None:
        # Using primary model
        lr_prediction, lr_confidence, processed_text = predict_sentiment(text, model, vectorizer, use_advanced)
        # Using gradient boosting model
        gb_prediction, gb_confidence, _ = predict_sentiment(text, gb_model, vectorizer, use_advanced)
        
        # Average the results (simple ensemble)
        if lr_prediction == gb_prediction:
            final_prediction = lr_prediction
            final_confidence = (lr_confidence + gb_confidence) / 2
        else:
            # If models disagree, use the one with higher confidence
            if lr_confidence > gb_confidence:
                final_prediction = lr_prediction
                final_confidence = lr_confidence
            else:
                final_prediction = gb_prediction
                final_confidence = gb_confidence
    
    # Use the primary model if available
    elif model is not None and vectorizer is not None:
        final_prediction, final_confidence, processed_text = predict_sentiment(text, model, vectorizer, use_advanced)
    
    # Use fallback if no models are loaded
    else:
        final_prediction, final_confidence, processed_text = fallback_predict_sentiment(text)
    
    sentiment = "positive" if final_prediction == 1 else "negative"
    
    # Return detailed response
    return jsonify({
        'sentiment': sentiment,
        'confidence': round(float(final_confidence), 2),
        'processed_text': processed_text,
        'input_text': text
    })

@app.route('/health', methods=['GET'])
def health_check():
    global model, vectorizer, gb_model
    
    # Check if models are loaded
    models_loaded = {
        'primary_model': model is not None,
        'vectorizer': vectorizer is not None,
        'gradient_boosting_model': gb_model is not None
    }
    
    # Test prediction on a simple sentence
    test_text = "This is a test message"
    try:
        if model is not None and vectorizer is not None:
            prediction, confidence, _ = predict_sentiment(test_text, model, vectorizer)
            prediction_works = True
        else:
            prediction_works = False
    except Exception:
        prediction_works = False
    
    return jsonify({
        'status': 'healthy' if models_loaded['primary_model'] and models_loaded['vectorizer'] else 'degraded',
        'models_loaded': models_loaded,
        'prediction_working': prediction_works,
        'version': '1.0.0'
    })

if __name__ == '__main__':
    # Load model and vectorizer before starting
    model, vectorizer = load_model_and_vectorizer(model_path, vectorizer_path)
    if model is None or vectorizer is None:
        print("Warning: Failed to load model or vectorizer. Will use fallback prediction.")
    else:
        print(f"Successfully loaded model from {model_path} and vectorizer from {vectorizer_path}")
    
    # Try to load gradient boosting model
    try:
        with open(gb_model_path, 'rb') as file:
            gb_model = pickle.load(file)
        print(f"Successfully loaded gradient boosting model from {gb_model_path}")
    except Exception as e:
        print(f"Warning: Failed to load gradient boosting model: {e}")
    
    # Run the Flask app
    port = int(os.environ.get('PORT', 5000))
    app.run(debug=True, host='0.0.0.0', port=port)
    