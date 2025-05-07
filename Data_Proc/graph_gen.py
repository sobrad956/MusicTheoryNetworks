import json
import os
import numpy as np
from gensim.models import Word2Vec, KeyedVectors
from gensim.utils import simple_preprocess
import networkx as nx
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import seaborn as sns
from sklearn.preprocessing import normalize
from community import community_louvain  # For community detection
from sentence_transformers import SentenceTransformer  # Add this import

# Download required NLTK data
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')  # Required for WordNet lemmatizer

# Get standard stopwords
standard_stopwords = set(stopwords.words('english'))

# Define important words to keep
important_words = {'not', 'no', 'only', 'also', 'as', 'but', 'nor', 'neither', 'either', 'too', 'very', 'so', 'such', 'than', 'that', 'this', 'these', 'those', 'then', 'just', 'more', 'most', 'less', 'least', 'much', 'many', 'few', 'fewer', 'fewest', 'little', 'less', 'least', 'more', 'most', 'much', 'many', 'some', 'any', 'all', 'both', 'each', 'every', 'either', 'neither', 'none', 'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine', 'ten', 'first', 'second', 'third', 'fourth', 'fifth', 'sixth', 'seventh', 'eighth', 'ninth', 'tenth'}

# Create custom stopwords set by removing important words
custom_stopwords = standard_stopwords - important_words

def load_publications_from_directory(directory_path):
    """Load publications from all JSON files in the directory."""
    # Load valid sources from article_sources.txt
    try:
        with open('article_sources.txt', 'r') as f:
            valid_sources = set(line.strip() for line in f)
    except FileNotFoundError:
        print("Error: article_sources.txt not found. Please run compute_edges.py first.")
        return []
    
    publications = []
    for filename in os.listdir(directory_path):
        if filename.endswith('.json'):
            # Check if this file's path is in valid_sources
            file_path = os.path.join(directory_path, filename)
            if file_path not in valid_sources:
                continue
                
            with open(file_path, 'r') as f:
                data = json.load(f)
                publications.append({
                    'title': data.get('title', ''),
                    'authors': data.get('authors', []),
                    'year': data.get('date', '')[:4] if data.get('date') else '',
                    'abstract': data.get('abstract', ''),
                    'doi': data.get('doi', ''),
                    'category': data.get('category', ''),
                    'paragraphs': data.get('paragraphs', []),
                    'source': file_path
                })
    return publications

def preprocess_abstract_word2vec(abstract):
    """Preprocess abstract for Word2Vec embeddings, removing stopwords except important ones."""
    if not abstract:
        return []
    
    # Tokenize
    tokens = word_tokenize(abstract.lower())
    
    # Remove stopwords (except important ones)
    tokens = [token for token in tokens if token not in custom_stopwords]
    
    # Lemmatize
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(token) for token in tokens]
    
    return tokens

def preprocess_abstract_bert(abstract):
    """Preprocess abstract for BERT embeddings, keeping all words."""
    if not abstract:
        return ""
    return abstract

def load_pretrained_model():
    """Load a pre-trained Word2Vec model."""
    # Try to load a pre-trained model, or download if not available
    try:
        # Try to load a pre-trained model
        model = KeyedVectors.load_word2vec_format('word2vec-google-news-300.bin', binary=True)
        print("Loaded pre-trained Word2Vec model")
    except:
        # If not available, download it
        print("Downloading pre-trained Word2Vec model...")
        import gensim.downloader
        model = gensim.downloader.load('word2vec-google-news-300')
        print("Downloaded pre-trained Word2Vec model")
    
    return model

def get_abstract_embedding(abstract, w2v_model):
    """Get Word2Vec embedding for an abstract."""
    if not abstract:
        return np.zeros(300)  # Default dimension for Word2Vec
    
    # Preprocess for Word2Vec
    tokens = preprocess_abstract_word2vec(abstract)
    if not tokens:
        return np.zeros(300)
    
    # Get embeddings for each word
    embeddings = []
    for token in tokens:
        try:
            embedding = w2v_model[token]
            embeddings.append(embedding)
        except KeyError:
            continue
    
    if not embeddings:
        return np.zeros(300)
    
    # Calculate weighted average
    embeddings = np.array(embeddings)
    weights = np.ones(len(embeddings))
    weighted_avg = np.average(embeddings, weights=weights, axis=0)
    
    # Normalize the final embedding
    return normalize([weighted_avg])[0]

def get_bert_embedding(abstract, model):
    """Get BERT embedding for an abstract."""
    if not abstract:
        return np.zeros(768)  # Default dimension for BERT
    
    # Preprocess for BERT (keeping all words)
    processed_abstract = preprocess_abstract_bert(abstract)
    return model.encode(processed_abstract, convert_to_numpy=True)

def create_similarity_graph(publications_data, std_devs=1):
    """Create a graph with edges that are more than specified standard deviations above mean similarity."""
    # First filter publications to only include articles
    filtered_publications = []
    for pub in publications_data:
        if pub.get('category', '').lower() == 'articles':
            filtered_publications.append(pub)
    
    print(f"Filtered out {len(publications_data) - len(filtered_publications)} non-article publications")
    print(f"Remaining articles: {len(filtered_publications)}")
    
    # Now get text for each article (abstract or first non-empty paragraph)
    articles_with_text = []
    no_abstract_count = 0     # Count of papers with no abstract
    
    print("\nArticles with no abstract:")
    for pub in filtered_publications:
        abstract = pub.get('abstract', '')
        if not abstract:
            no_abstract_count += 1
            # Print title and source for articles with no abstract
            print(f"  Title: {pub.get('title', 'No title')}")
            print(f"  Source: {pub.get('source', 'No source')}")
            print()
            
            # Try to get first non-empty paragraph from paragraphs
            paragraphs = pub.get('paragraphs', [])
            for paragraph in paragraphs:
                if isinstance(paragraph, dict):
                    text = paragraph.get('text', '')
                    if text.strip():  # Check if text is not empty after stripping whitespace
                        abstract = text
                        break
                elif isinstance(paragraph, str) and paragraph.strip():
                    abstract = paragraph
                    break
        
        # Add to articles_with_text if we found either an abstract or first paragraph
        if abstract:
            articles_with_text.append(pub)
    
    print(f"Found {no_abstract_count} articles with no abstract (using first paragraph instead)")
    print(f"Articles with text (abstract or first paragraph): {len(articles_with_text)}")
    
    # Extract text (abstract or first paragraph) for articles with text
    texts = []
    for pub in articles_with_text:
        abstract = pub.get('abstract', '')
        if not abstract:
            paragraphs = pub.get('paragraphs', [])
            for paragraph in paragraphs:
                if isinstance(paragraph, dict):
                    text = paragraph.get('text', '')
                    if text.strip():
                        abstract = text
                        break
                elif isinstance(paragraph, str) and paragraph.strip():
                    abstract = paragraph
                    break
        texts.append(abstract)
    
    # Load pre-trained Word2Vec model
    w2v_model = load_pretrained_model()
    
    # Calculate embeddings for all texts
    text_embeddings = [get_abstract_embedding(text, w2v_model) 
                      for text in texts]
    
    # Calculate pairwise cosine similarities
    similarities = cosine_similarity(text_embeddings)
    
    # Calculate mean and standard deviation of similarities
    mean_similarity = np.mean(similarities)
    std_similarity = np.std(similarities)
    threshold = mean_similarity + std_devs * std_similarity
    
    print(f"Mean similarity: {mean_similarity:.3f}")
    print(f"Standard deviation: {std_similarity:.3f}")
    print(f"Similarity threshold ({std_devs} std devs): {threshold:.3f}")
    
    # Create graph
    G = nx.Graph()
    
    # Add all articles as nodes (including those without text)
    for i, pub in enumerate(filtered_publications):
        # Convert authors list to string to ensure serializability
        authors_str = ', '.join(pub.get('authors', []))
        G.add_node(i, 
                  title=str(pub.get('title', '')),
                  authors=authors_str,
                  year=str(pub.get('year', '')),
                  doi=str(pub.get('doi', '')),
                  category=str(pub.get('category', '')),
                  has_text=pub in articles_with_text)
    
    # Add edges with similarity weights only if above threshold (only for articles with text)
    edge_count = 0
    for i in range(len(articles_with_text)):
        for j in range(i + 1, len(articles_with_text)):
            # Find the indices in the full publication list
            i_full = filtered_publications.index(articles_with_text[i])
            j_full = filtered_publications.index(articles_with_text[j])
            # Add edge only if similarity is above threshold
            weight = float(similarities[i][j])
            if weight > threshold:
                G.add_edge(i_full, j_full, weight=weight)
                edge_count += 1
    
    print(f"Number of edges above threshold: {edge_count}")
    print(f"Number of isolated nodes (no text): {len(filtered_publications) - len(articles_with_text)}")
    
    # Return both the graph, the full similarity matrix, and the articles_with_text list
    return G, similarities, articles_with_text

def create_similarity_graph_bert(publications_data, std_devs=1):
    """Create a graph using BERT embeddings for abstract similarity."""
    # First filter publications to only include articles
    filtered_publications = []
    for pub in publications_data:
        if pub.get('category', '').lower() == 'articles':
            filtered_publications.append(pub)
    
    print(f"Filtered out {len(publications_data) - len(filtered_publications)} non-article publications")
    print(f"Remaining articles: {len(filtered_publications)}")
    
    # Now get text for each article (abstract or first non-empty paragraph)
    articles_with_text = []
    no_abstract_count = 0     # Count of papers with no abstract
    
    print("\nArticles with no abstract:")
    for pub in filtered_publications:
        abstract = pub.get('abstract', '')
        if not abstract:
            no_abstract_count += 1
            # Print title and source for articles with no abstract
            print(f"  Title: {pub.get('title', 'No title')}")
            print(f"  Source: {pub.get('source', 'No source')}")
            print()
            
            # Try to get first non-empty paragraph from paragraphs
            paragraphs = pub.get('paragraphs', [])
            for paragraph in paragraphs:
                if isinstance(paragraph, dict):
                    text = paragraph.get('text', '')
                    if text.strip():  # Check if text is not empty after stripping whitespace
                        abstract = text
                        break
                elif isinstance(paragraph, str) and paragraph.strip():
                    abstract = paragraph
                    break
        
        # Add to articles_with_text if we found either an abstract or first paragraph
        if abstract:
            articles_with_text.append(pub)
    
    print(f"Found {no_abstract_count} articles with no abstract (using first paragraph instead)")
    print(f"Articles with text (abstract or first paragraph): {len(articles_with_text)}")
    
    # Extract text (abstract or first paragraph) for articles with text
    texts = []
    for pub in articles_with_text:
        abstract = pub.get('abstract', '')
        if not abstract:
            paragraphs = pub.get('paragraphs', [])
            for paragraph in paragraphs:
                if isinstance(paragraph, dict):
                    text = paragraph.get('text', '')
                    if text.strip():
                        abstract = text
                        break
                elif isinstance(paragraph, str) and paragraph.strip():
                    abstract = paragraph
                    break
        texts.append(abstract)
    
    # Load BERT model
    print("Loading BERT model...")
    bert_model = SentenceTransformer('all-MiniLM-L6-v2')
    
    # Calculate BERT embeddings for all texts
    print("Calculating BERT embeddings...")
    text_embeddings = [get_bert_embedding(text, bert_model) for text in texts]
    
    # Calculate pairwise cosine similarities
    similarities = cosine_similarity(text_embeddings)
    
    # Calculate mean and standard deviation of similarities
    mean_similarity = np.mean(similarities)
    std_similarity = np.std(similarities)
    threshold = mean_similarity + std_devs * std_similarity
    
    print(f"Mean similarity: {mean_similarity:.3f}")
    print(f"Standard deviation: {std_similarity:.3f}")
    print(f"Similarity threshold ({std_devs} std devs): {threshold:.3f}")
    
    # Create graph
    G = nx.Graph()
    
    # Add all articles as nodes (including those without text)
    for i, pub in enumerate(filtered_publications):
        # Convert authors list to string to ensure serializability
        authors_str = ', '.join(pub.get('authors', []))
        G.add_node(i, 
                  title=str(pub.get('title', '')),
                  authors=authors_str,
                  year=str(pub.get('year', '')),
                  doi=str(pub.get('doi', '')),
                  category=str(pub.get('category', '')),
                  has_text=pub in articles_with_text)
    
    # Add edges with similarity weights only if above threshold (only for articles with text)
    edge_count = 0
    for i in range(len(articles_with_text)):
        for j in range(i + 1, len(articles_with_text)):
            # Find the indices in the full publication list
            i_full = filtered_publications.index(articles_with_text[i])
            j_full = filtered_publications.index(articles_with_text[j])
            # Add edge only if similarity is above threshold
            weight = float(similarities[i][j])
            if weight > threshold:
                G.add_edge(i_full, j_full, weight=weight)
                edge_count += 1
    
    print(f"Number of edges above threshold: {edge_count}")
    print(f"Number of isolated nodes (no text): {len(filtered_publications) - len(articles_with_text)}")
    
    # Return both the graph, the full similarity matrix, and the articles_with_text list
    return G, similarities, articles_with_text

def visualize_graph(G, publications_data, layout='spring', title_suffix='', output_dir='res'):
    """Create a visualization of the graph."""
    plt.figure(figsize=(20, 20))
    
    # Set up the layout with adjusted parameters
    if layout == 'spring':
        pos = nx.spring_layout(G, 
                             k=2,           # Increase node spacing
                             iterations=100, # More iterations for better layout
                             seed=42)       # Fixed seed for reproducibility
    elif layout == 'fruchterman_reingold':
        pos = nx.fruchterman_reingold_layout(G, 
                                            k=2,           # Increase node spacing
                                            iterations=100, # More iterations for better layout
                                            seed=42)       # Fixed seed for reproducibility
    else:
        pos = nx.spring_layout(G, seed=42)
    
    # Get edge weights and normalize them for opacity
    edge_weights = [G[u][v]['weight'] for u, v in G.edges()]
    if edge_weights:
        min_weight = min(edge_weights)
        max_weight = max(edge_weights)
        # Normalize weights to [0.1, 1.0] range for opacity
        normalized_weights = [(w - min_weight) / (max_weight - min_weight) * 0.9 + 0.1 
                            for w in edge_weights]
    else:
        normalized_weights = []
    
    # Draw edges with weights as width and opacity
    nx.draw_networkx_edges(G, pos, 
                          width=[w * 0.5 for w in edge_weights],  # Thinner edges
                          alpha=normalized_weights,                # Opacity based on normalized weight
                          edge_color='gray')
    
    # Draw nodes
    nx.draw_networkx_nodes(G, pos, 
                          node_size=50,           # Smaller nodes
                          node_color='lightblue',
                          alpha=0.7)              # Slightly transparent nodes
    
    plt.title(f"Publication Similarity Network{title_suffix}\n(Edge opacity and width based on abstract similarity)")
    plt.axis('off')
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Save the plot
    plt.savefig(os.path.join(output_dir, f'publication_similarity_network_{layout}{title_suffix}.png'), 
                dpi=300, 
                bbox_inches='tight',
                facecolor='white')
    plt.close()

def visualize_graph_with_communities(G, publications_data, layout='spring', title_suffix='', output_dir='res'):
    """Create a visualization of the graph with communities highlighted."""
    plt.figure(figsize=(20, 20))
    
    # Detect communities using Louvain method
    communities = community_louvain.best_partition(G)
    
    # Count number of communities
    num_communities = len(set(communities.values()))
    print(f"Number of communities detected: {num_communities}")
    
    # Set up the layout with adjusted parameters
    if layout == 'spring':
        pos = nx.spring_layout(G, 
                             k=2,           # Increase node spacing
                             iterations=100, # More iterations for better layout
                             seed=42)       # Fixed seed for reproducibility
    elif layout == 'fruchterman_reingold':
        pos = nx.fruchterman_reingold_layout(G, 
                                            k=2,           # Increase node spacing
                                            iterations=100, # More iterations for better layout
                                            seed=42)       # Fixed seed for reproducibility
    else:
        pos = nx.spring_layout(G, seed=42)
    
    # Get edge weights and normalize them for opacity
    edge_weights = [G[u][v]['weight'] for u, v in G.edges()]
    if edge_weights:
        min_weight = min(edge_weights)
        max_weight = max(edge_weights)
        # Normalize weights to [0.1, 1.0] range for opacity
        normalized_weights = [(w - min_weight) / (max_weight - min_weight) * 0.9 + 0.1 
                            for w in edge_weights]
    else:
        normalized_weights = []
    
    # Draw edges with weights as width and opacity
    nx.draw_networkx_edges(G, pos, 
                          width=[w * 0.5 for w in edge_weights],  # Thinner edges
                          alpha=normalized_weights,                # Opacity based on normalized weight
                          edge_color='gray')
    
    # Draw nodes with colors based on community
    node_colors = [communities[node] for node in G.nodes()]
    nx.draw_networkx_nodes(G, pos, 
                          node_size=50,           # Smaller nodes
                          node_color=node_colors,
                          cmap=plt.cm.tab20,      # Use a colormap with many colors
                          alpha=0.7)              # Slightly transparent nodes
    
    plt.title(f"Publication Similarity Network with Communities{title_suffix}\n(Edge opacity and width based on abstract similarity)")
    plt.axis('off')
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Save the plot
    plt.savefig(os.path.join(output_dir, f'publication_similarity_network_communities_{layout}{title_suffix}.png'), 
                dpi=300, 
                bbox_inches='tight',
                facecolor='white')
    plt.close()
    
    return communities

def plot_edge_weight_distribution(all_similarities, threshold, title_suffix='', output_dir='res'):
    """Create a histogram of all edge weights."""
    # Flatten the similarity matrix and remove self-similarities (diagonal)
    flat_similarities = all_similarities[np.triu_indices_from(all_similarities, k=1)]
    
    plt.figure(figsize=(10, 6))
    plt.hist(flat_similarities, bins=50, edgecolor='black')
    plt.title(f'Distribution of All Edge Weights (Cosine Similarities){title_suffix}')
    plt.xlabel('Similarity Score')
    plt.ylabel('Number of Edges')
    plt.grid(True, alpha=0.3)
    
    # Add vertical line at threshold
    plt.axvline(x=threshold, color='r', linestyle='--', 
                label=f'Threshold (mean + {threshold - np.mean(flat_similarities):.1f} std = {threshold:.3f})')
    plt.legend()
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Save the plot
    plt.savefig(os.path.join(output_dir, f'edge_weight_distribution{title_suffix}.png'), 
                dpi=300, 
                bbox_inches='tight',
                facecolor='white')
    plt.close()

def main():
    # Create main results directory
    os.makedirs('res', exist_ok=True)
    
    # Load publications from the data directory
    publications = load_publications_from_directory('mto/data')
    
    # Create subdirectories for different embedding methods
    res_word2vec = os.path.join('res', 'word2vec_1_std_dev')
    res_word2vec_all = os.path.join('res', 'word2vec_all_edges')
    res_bert = os.path.join('res', 'bert_1_std_dev')
    res_bert_all = os.path.join('res', 'bert_all_edges')
    os.makedirs(res_word2vec, exist_ok=True)
    os.makedirs(res_word2vec_all, exist_ok=True)
    os.makedirs(res_bert, exist_ok=True)
    os.makedirs(res_bert_all, exist_ok=True)
    
    # Create and analyze Word2Vec-based graph
    print("\n--- Creating Word2Vec-based graph ---")
    graph_word2vec, similarities_word2vec, articles_with_text_word2vec = create_similarity_graph(publications, std_devs=1)
    
    # Save Word2Vec graph with thresholded edges
    word2vec_gexf_path = os.path.join(res_word2vec, "publication_similarity_graph.gexf")
    nx.write_gexf(graph_word2vec, word2vec_gexf_path, encoding='utf-8')
    print(f"\nWord2Vec graph (thresholded) saved with {graph_word2vec.number_of_nodes()} nodes and {graph_word2vec.number_of_edges()} edges")
    
    # Create visualizations for Word2Vec graph with thresholded edges
    visualize_graph(graph_word2vec, publications, layout='spring', title_suffix='_word2vec', output_dir=res_word2vec)
    visualize_graph(graph_word2vec, publications, layout='fruchterman_reingold', title_suffix='_word2vec', output_dir=res_word2vec)
    visualize_graph_with_communities(graph_word2vec, publications, layout='spring', title_suffix='_word2vec', output_dir=res_word2vec)
    visualize_graph_with_communities(graph_word2vec, publications, layout='fruchterman_reingold', title_suffix='_word2vec', output_dir=res_word2vec)
    
    # Calculate threshold for Word2Vec distribution plot
    mean_similarity = np.mean(similarities_word2vec)
    std_similarity = np.std(similarities_word2vec)
    threshold = mean_similarity + std_similarity
    plot_edge_weight_distribution(similarities_word2vec, threshold, title_suffix='_word2vec', output_dir=res_word2vec)
    
    # Create and save Word2Vec graph with all edges
    print("\n--- Creating Word2Vec-based graph with all edges ---")
    graph_word2vec_all = nx.Graph()
    # Copy all nodes
    for node, data in graph_word2vec.nodes(data=True):
        graph_word2vec_all.add_node(node, **data)
    # Add all possible edges
    for i in range(len(publications)):
        for j in range(i + 1, len(publications)):
            if publications[i] in articles_with_text_word2vec and publications[j] in articles_with_text_word2vec:
                i_idx = articles_with_text_word2vec.index(publications[i])
                j_idx = articles_with_text_word2vec.index(publications[j])
                graph_word2vec_all.add_edge(i, j, weight=float(similarities_word2vec[i_idx][j_idx]))
    
    # Save Word2Vec graph with all edges
    word2vec_all_gexf_path = os.path.join(res_word2vec_all, "publication_similarity_graph_all_edges.gexf")
    nx.write_gexf(graph_word2vec_all, word2vec_all_gexf_path, encoding='utf-8')
    print(f"\nWord2Vec graph (all edges) saved with {graph_word2vec_all.number_of_nodes()} nodes and {graph_word2vec_all.number_of_edges()} edges")
    
    # Create visualizations for Word2Vec graph with all edges
    visualize_graph(graph_word2vec_all, publications, layout='spring', title_suffix='_word2vec_all_edges', output_dir=res_word2vec_all)
    visualize_graph(graph_word2vec_all, publications, layout='fruchterman_reingold', title_suffix='_word2vec_all_edges', output_dir=res_word2vec_all)
    visualize_graph_with_communities(graph_word2vec_all, publications, layout='spring', title_suffix='_word2vec_all_edges', output_dir=res_word2vec_all)
    visualize_graph_with_communities(graph_word2vec_all, publications, layout='fruchterman_reingold', title_suffix='_word2vec_all_edges', output_dir=res_word2vec_all)
    
    # Create and analyze BERT-based graph
    print("\n--- Creating BERT-based graph ---")
    graph_bert, similarities_bert, articles_with_text_bert = create_similarity_graph_bert(publications, std_devs=1)
    
    # Save BERT graph with thresholded edges
    bert_gexf_path = os.path.join(res_bert, "publication_similarity_graph.gexf")
    nx.write_gexf(graph_bert, bert_gexf_path, encoding='utf-8')
    print(f"\nBERT graph (thresholded) saved with {graph_bert.number_of_nodes()} nodes and {graph_bert.number_of_edges()} edges")
    
    # Create visualizations for BERT graph with thresholded edges
    visualize_graph(graph_bert, publications, layout='spring', title_suffix='_bert', output_dir=res_bert)
    visualize_graph(graph_bert, publications, layout='fruchterman_reingold', title_suffix='_bert', output_dir=res_bert)
    visualize_graph_with_communities(graph_bert, publications, layout='spring', title_suffix='_bert', output_dir=res_bert)
    visualize_graph_with_communities(graph_bert, publications, layout='fruchterman_reingold', title_suffix='_bert', output_dir=res_bert)
    
    # Calculate threshold for BERT distribution plot
    mean_similarity = np.mean(similarities_bert)
    std_similarity = np.std(similarities_bert)
    threshold = mean_similarity + std_similarity
    plot_edge_weight_distribution(similarities_bert, threshold, title_suffix='_bert', output_dir=res_bert)
    
    # Create and save BERT graph with all edges
    print("\n--- Creating BERT-based graph with all edges ---")
    graph_bert_all = nx.Graph()
    # Copy all nodes
    for node, data in graph_bert.nodes(data=True):
        graph_bert_all.add_node(node, **data)
    # Add all possible edges
    for i in range(len(publications)):
        for j in range(i + 1, len(publications)):
            if publications[i] in articles_with_text_bert and publications[j] in articles_with_text_bert:
                i_idx = articles_with_text_bert.index(publications[i])
                j_idx = articles_with_text_bert.index(publications[j])
                graph_bert_all.add_edge(i, j, weight=float(similarities_bert[i_idx][j_idx]))
    
    # Save BERT graph with all edges
    bert_all_gexf_path = os.path.join(res_bert_all, "publication_similarity_graph_all_edges.gexf")
    nx.write_gexf(graph_bert_all, bert_all_gexf_path, encoding='utf-8')
    print(f"\nBERT graph (all edges) saved with {graph_bert_all.number_of_nodes()} nodes and {graph_bert_all.number_of_edges()} edges")
    
    # Create visualizations for BERT graph with all edges
    visualize_graph(graph_bert_all, publications, layout='spring', title_suffix='_bert_all_edges', output_dir=res_bert_all)
    visualize_graph(graph_bert_all, publications, layout='fruchterman_reingold', title_suffix='_bert_all_edges', output_dir=res_bert_all)
    visualize_graph_with_communities(graph_bert_all, publications, layout='spring', title_suffix='_bert_all_edges', output_dir=res_bert_all)
    visualize_graph_with_communities(graph_bert_all, publications, layout='fruchterman_reingold', title_suffix='_bert_all_edges', output_dir=res_bert_all)
    
    # Print final summary
    print("\nFinal Summary:")
    print(f"Word2Vec graph (thresholded): {graph_word2vec.number_of_nodes()} nodes, {graph_word2vec.number_of_edges()} edges")
    print(f"Word2Vec graph (all edges): {graph_word2vec_all.number_of_nodes()} nodes, {graph_word2vec_all.number_of_edges()} edges")
    print(f"BERT graph (thresholded): {graph_bert.number_of_nodes()} nodes, {graph_bert.number_of_edges()} edges")
    print(f"BERT graph (all edges): {graph_bert_all.number_of_nodes()} nodes, {graph_bert_all.number_of_edges()} edges")

if __name__ == "__main__":
    main()