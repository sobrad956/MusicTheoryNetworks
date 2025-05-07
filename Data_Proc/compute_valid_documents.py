import json
import os
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import nltk
from gensim.models import KeyedVectors
import networkx as nx
import matplotlib.pyplot as plt
import seaborn as sns
import gzip
import shutil
import requests
from tqdm import tqdm

# Download required NLTK data
nltk.download('punkt')
nltk.download('stopwords')

# Define important words to keep
important_words = {'not', 'no', 'only', 'also', 'as'}
custom_stopwords = set(stopwords.words('english')) - important_words

def download_word2vec_model():
    """Download the pretrained Word2Vec model if it doesn't exist"""
    model_path = 'GoogleNews-vectors-negative300.bin'
    if not os.path.exists(model_path):
        print("Downloading pretrained Word2Vec model...")
        
        # Use a direct download link from a reliable source
        url = 'https://s3.amazonaws.com/dl4j-distribution/GoogleNews-vectors-negative300.bin.gz'
        
        # Download the file with progress bar
        response = requests.get(url, stream=True)
        total_size = int(response.headers.get('content-length', 0))
        
        with open(model_path + '.gz', 'wb') as f, tqdm(
            desc='Downloading',
            total=total_size,
            unit='iB',
            unit_scale=True,
            unit_divisor=1024,
        ) as pbar:
            for data in response.iter_content(chunk_size=1024):
                size = f.write(data)
                pbar.update(size)
        
        # Extract the gzipped file
        print("Extracting model file...")
        with gzip.open(model_path + '.gz', 'rb') as f_in:
            with open(model_path, 'wb') as f_out:
                shutil.copyfileobj(f_in, f_out)
        
        # Remove the gzipped file
        os.remove(model_path + '.gz')
        print("Model downloaded and extracted successfully!")
    return model_path

def preprocess_text(text):
    """Preprocess text for embedding"""
    if not text:
        return []
    # Tokenize and convert to lowercase
    tokens = word_tokenize(text.lower())
    # Remove stopwords except important ones
    tokens = [token for token in tokens if token not in custom_stopwords]
    return tokens

def get_text_embedding(text, model):
    """Get embedding for text using Word2Vec model"""
    tokens = preprocess_text(text)
    if not tokens:
        return np.zeros(model.vector_size)
    
    # Get embeddings for all words
    embeddings = [model[token] for token in tokens if token in model]
    if not embeddings:
        return np.zeros(model.vector_size)
    
    # Return average embedding
    return np.mean(embeddings, axis=0)

def get_abstract_from_source(source_path):
    """Read abstract or first paragraph from the source JSON file"""
    try:
        with open(source_path, 'r') as f:
            data = json.load(f)
            
            # Try to get abstract first
            abstract = data.get('abstract', '')
            
            # If no abstract, try to get the first paragraph from the content
            if not abstract:
                content = data.get('content', [])
                if content and isinstance(content, list) and len(content) > 0:
                    # Get the first paragraph from the content
                    first_paragraph = content[0]
                    if isinstance(first_paragraph, dict):
                        # If it's a dictionary, try to get the text
                        abstract = first_paragraph.get('text', '')
                    elif isinstance(first_paragraph, str):
                        # If it's a string, use it directly
                        abstract = first_paragraph
            
            # Return the text and data if we found either an abstract or first paragraph
            if abstract:
                return abstract, data
            return '', None
            
    except (FileNotFoundError, json.JSONDecodeError) as e:
        print(f"Warning: Could not read {source_path}: {str(e)}")
        return '', None

def plot_similarity_distribution(similarities, threshold):
    """Plot the distribution of cosine similarities"""
    # Flatten the similarity matrix (excluding diagonal)
    flat_similarities = similarities[np.triu_indices_from(similarities, k=1)]
    
    plt.figure(figsize=(10, 6))
    sns.histplot(flat_similarities, bins=50, kde=True)
    plt.axvline(x=threshold, color='r', linestyle='--', label=f'Threshold: {threshold:.3f}')
    plt.title('Distribution of Cosine Similarities Between Articles')
    plt.xlabel('Cosine Similarity')
    plt.ylabel('Count')
    plt.legend()
    plt.savefig('similarity_distribution.png')
    plt.close()

def create_similarity_network(embeddings, article_ids, threshold):
    """Create a network where nodes are articles and edges exist if similarity > threshold"""
    G = nx.Graph()
    
    # Add nodes
    for article_id in article_ids:
        G.add_node(article_id)
    
    # Add edges based on similarity threshold
    for i in range(len(embeddings)):
        for j in range(i+1, len(embeddings)):
            similarity = cosine_similarity([embeddings[i]], [embeddings[j]])[0][0]
            if similarity > threshold:
                G.add_edge(article_ids[i], article_ids[j], weight=similarity)
    
    return G

def count_publications():
    try:
        # Get absolute path to mto_library.json
        current_dir = os.path.dirname(os.path.abspath(__file__))
        library_path = os.path.join(current_dir, 'mto', 'mto_library.json')
        
        # Load the JSON file
        with open(library_path, 'r') as f:
            data = json.load(f)
        
        # Check if the data is a dictionary and has a 'publications' key
        if isinstance(data, dict) and 'publications' in data:
            publications = data['publications']
            # Count only publications where category is 'articles'
            article_count = sum(1 for pub_id, pub_data in publications.items() 
                              if pub_data.get('data', {}).get('category', '').lower() == 'articles')
            print(f"Total number of publications: {len(publications)}")
            print(f"Number of articles: {article_count}")
            
            # Create and save list of sources
            sources = []
            for pub_id, pub_data in publications.items():
                if pub_data.get('data', {}).get('category', '').lower() == 'articles':
                    source_path = pub_data.get('source', '')
                    if source_path:
                        sources.append(source_path)
            
            # Save sources to file
            with open('article_sources.txt', 'w') as f:
                for source in sources:
                    f.write(f"{source}\n")
            
            print(f"Number of sources saved: {len(sources)}")
            
            # Get text for all articles
            texts = []
            article_ids = []
            for pub_id, pub_data in publications.items():
                if pub_data.get('data', {}).get('category', '').lower() == 'articles':
                    source_path = pub_data.get('source', '')
                    if source_path:
                        text, _ = get_abstract_from_source(source_path)
                        if text:  # This will be true if we found either an abstract or first paragraph
                            texts.append(text)
                            article_ids.append(pub_id)
            
            print(f"Number of articles with text (abstract or first paragraph): {len(texts)}")
            
        else:
            print("The JSON file does not contain a 'publications' field")
            
    except FileNotFoundError:
        print("Error: mto_library.json file not found")
    except json.JSONDecodeError:
        print("Error: Invalid JSON format in mto_library.json")
    except Exception as e:
        print(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    count_publications() 